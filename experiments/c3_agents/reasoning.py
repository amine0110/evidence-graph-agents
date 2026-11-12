"""C3 systems S2, S3 and S4 on the stored evidence graphs, served by one LLM.

  S3  graph: the reasoning agent reads the clip's graph through tools (time
      window, concept search, edges of a node, provenance) and cites nodes.
  S2  messages: each perception agent (video, audio_events, speech) is the
      same LLM given only its own outputs; it sends a report, and the
      reasoning agent may ask each agent follow-up questions. No graph, no
      edges, no node ids shared between agents.
  S4  flat text: every node of every agent as one time-sorted text, one pass.

All three end with the same final turn: the task question of task.py. The
score is P(Yes) / (P(Yes) + P(No)) on the first answer token, as for S1.

Ablations of S3 (--drop / --no-edges / --no-confidence) and the faithfulness
reruns (--delete-cited / --delete-random from an earlier run) change only
which nodes or fields the agent can see. The graphs root must not contain
clips.csv: clip ids are neutral and nothing here may map them back to files.

Talks to an OpenAI-compatible server (vLLM) at --server; clips run concurrently.
"""

import argparse
import json
import math
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evgraph import EdgeType
from audio_events import clip_graph
from edges import LEXICON, concepts
from task import QUESTION, TASK_VERSION, parse_answer

ALIASES = {"shooting": "gun", "gunshot": "gun", "gunfire": "gun", "weapon": "gun", "firearm": "gun",
           "explosion": "fire", "smoke": "fire", "fighting": "fight", "violence": "fight", "assault": "fight",
           "abuse": "fight", "accident": "crash", "collision": "crash", "riot": "crowd", "protest": "crowd",
           "police": "emergency", "blood": "injury", "scream": "distress", "car": "vehicle"}
MAX_STEPS = 8
BIN = 10
VIOLENT = {"gun", "knife", "fire", "fight", "crash", "injury", "distress", "threat"}
# S3 v2 was developed on the dev set only (dev_set.py), in four attempts; "c" had the best dev F1
# and is the one run on the test set. a, b: earlier attempts (see the change log); d: full flat view.
S3V2_VARIANTS = ("c", "d")
MAX_QUESTIONS = 3
MAX_STEPS_V2 = 12
LIST_LIMIT = 40
CITE_LINE = "On a third line, list the ids of the evidence items your answer rests on, like n3, n17, or write none."
AGENTS = {"video": ("video",), "audio_events": ("audio_events",), "speech": ("speech", "speech_tags")}


# --- talking to the server ---------------------------------------------------

class LLM:
    def __init__(self, server, model, max_tokens=400, no_think=False):
        self.url, self.model, self.max_tokens = server.rstrip("/") + "/v1/chat/completions", model, max_tokens
        self.no_think = no_think

    def chat(self, messages, max_tokens=None, logprobs=False):
        body = {"model": self.model, "messages": messages, "temperature": 0.0,
                "max_tokens": max_tokens or self.max_tokens}
        if logprobs:
            body.update(logprobs=True, top_logprobs=20)
        if self.no_think:  # models with a thinking mode answer directly, like Gemma
            body["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    out = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 400:  # not transient: usually the prompt exceeds the context
                    raise RuntimeError(f"server refused: {e.read().decode(errors='ignore')[:300]}") from None
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        choice = out["choices"][0]
        return choice["message"]["content"] or "", out.get("usage", {}), choice.get("logprobs")


def yes_probability(logprobs):
    """P(Yes) / (P(Yes) + P(No)) from the first generated token's top log probabilities."""
    if not logprobs or not logprobs.get("content"):
        return None
    yes = no = 0.0
    for cand in logprobs["content"][0]["top_logprobs"]:
        word = cand["token"].strip().lower()
        if word == "yes":
            yes += math.exp(cand["logprob"])
        elif word == "no":
            no += math.exp(cand["logprob"])
    return yes / (yes + no) if yes + no > 0 else None


class Usage:
    def __init__(self):
        self.prompt = self.completion = self.calls = 0

    def add(self, usage):
        self.prompt += usage.get("prompt_tokens", 0)
        self.completion += usage.get("completion_tokens", 0)
        self.calls += 1

    def as_dict(self):
        return {"prompt_tokens": self.prompt, "completion_tokens": self.completion, "calls": self.calls}


# --- what the agent may see ------------------------------------------------------

class View:
    """The nodes and edges of one clip as the agent sees them, with short ids n1, n2, ..."""

    def __init__(self, graph, drop=(), delete=(), edges=True, confidence=True):
        drop = {p for name in drop for p in AGENTS.get(name, (name,))}
        self.graph, self.edges, self.confidence = graph, edges, confidence
        nodes = [n for n in graph.nodes.values()
                 if n.source.producer not in drop and n.id not in set(delete)]
        self.nodes = sorted(nodes, key=lambda n: (n.t_obs[0], n.t_obs[1], n.id))
        self.short = {n.id: f"n{k + 1}" for k, n in enumerate(self.nodes)}
        self.long = {v: k for k, v in self.short.items()}
        self.duration = max((n.t_obs[1] for n in graph.nodes.values()), default=0.0)
        self.shown = set()

    def line(self, n):
        c, conf = n.content, f", conf {n.confidence:.2f}" if self.confidence else ""
        t = f"{n.t_obs[0]:.0f}-{n.t_obs[1]:.0f}s"
        if n.kind == "caption":
            body = f"video description: {c['text']}"
        elif n.kind == "detection":
            body = f"video detector saw '{c['object']}' (up to {c['max_count']} at once, {c['seconds_seen']} s{conf})"
        elif n.kind == "sound_event":
            body = f"sound: {c['event']} ({c['top_class']}{conf})"
        elif n.kind == "transcript_segment":
            body = f"speech ({c['language']}{conf}): \"{c['text'].strip()}\""
        elif n.kind == "speech_tag":
            body = f"speech tagged {c['tag']}{conf}: \"{c['text'].strip()}\""
        else:
            body = f"{n.kind}: {json.dumps(c, ensure_ascii=False)}"
        return f"[{self.short[n.id]}] {t} {body}"

    def lines(self, nodes, limit=LIST_LIMIT):
        nodes = list(nodes)
        self.shown.update(n.id for n in nodes[:limit])
        text = "\n".join(self.line(n) for n in nodes[:limit])
        if len(nodes) > limit:
            text += f"\n... {len(nodes) - limit} more; ask for a narrower window"
        return text or "nothing"

    def visible(self, node_id):
        return node_id in self.short

    # tools (S3)

    def overview(self):
        counts = {}
        for n in self.nodes:
            counts[n.kind] = counts.get(n.kind, 0) + 1
        cues = [n for n in self.nodes if concepts(n)[0] and n.kind != "caption"]
        cues = sorted(cues, key=lambda n: -n.confidence)[:15] if self.confidence else cues[:15]
        return (f"Clip length {self.duration:.0f} s. Evidence items: "
                + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
                + ".\nStrongest non-caption cues:\n" + self.lines(sorted(cues, key=lambda n: n.t_obs[0])))

    # S3 v2: whole-clip timeline of concepts and corroborated events built from the edges

    AGENT_NAME = {"video": "video", "audio_events": "sound", "speech": "speech", "speech_tags": "speech"}

    def timeline(self, step=BIN):
        """One line per time bin with the concepts each agent asserts in it."""
        bins = {}
        for n in self.nodes:
            pos = concepts(n)[0] - {"vehicle"}
            if not pos:
                continue
            who = "detector" if n.kind == "detection" else self.AGENT_NAME.get(n.source.producer, n.source.producer)
            for b in range(int(n.t_obs[0] // step), int(max(n.t_obs[1] - 1e-6, n.t_obs[0]) // step) + 1):
                slot = bins.setdefault(b, {}).setdefault(who, {})
                for c in pos:
                    slot[c] = max(slot.get(c, 0.0), n.confidence)
        for slots in bins.values():  # a detector concept is shown only if another agent asserts it too
            det = slots.get("detector", {})
            others = {c for who, found in slots.items() if who != "detector" for c in found}
            for c in [c for c in det if c not in others]:
                del det[c]
            if "detector" in slots and not det:
                del slots["detector"]
        self._bins = bins
        lines = []
        for b in range(int(self.duration // step) + 1):
            parts = []
            for who in ("video", "detector", "sound", "speech"):
                found = bins.get(b, {}).get(who)
                if found:
                    items = ", ".join(f"{c} {v:.2f}" if self.confidence else c for c, v in sorted(found.items()))
                    parts.append(f"{who}: {items}")
            lines.append(f"{b * step}-{min((b + 1) * step, self.duration):.0f}s  " + ("; ".join(parts) or "-"))
        return "\n".join(lines)

    def event_groups(self):
        """Groups of nodes from at least two agents joined by supports edges, strongest first."""
        if not self.edges:
            return []
        parent = {n.id: n.id for n in self.nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for n in self.nodes:
            for edge, other in self.graph.neighbours(n.id, [EdgeType.SUPPORTS]):
                if other.id in parent:
                    parent[find(n.id)] = find(other.id)
        groups = {}
        for n in self.nodes:
            groups.setdefault(find(n.id), []).append(n)
        out = []
        for members in groups.values():
            agents = {self.AGENT_NAME.get(m.source.producer, m.source.producer) if m.kind != "detection" else "detector"
                      for m in members}
            if len(agents - {"detector"}) < 2:
                continue
            strength = sum(sorted((m.confidence for m in members), reverse=True)[:3])
            out.append((strength, members, agents))
        return sorted(out, key=lambda g: -g[0])

    def events(self, limit=8):
        groups = self.event_groups()
        if not groups:
            return "no corroborated events" if self.edges else "links are not available"
        lines = []
        for k, (strength, members, agents) in enumerate(groups[:limit], 1):
            start = min(m.t_obs[0] for m in members); end = max(m.t_obs[1] for m in members)
            found = sorted({c for m in members for c in concepts(m)[0]})
            ids = ", ".join(self.short[m.id] for m in sorted(members, key=lambda m: -m.confidence)[:6])
            lines.append(f"event {k}: {start:.0f}-{end:.0f}s, agents {', '.join(sorted(agents))}, "
                         f"concepts {', '.join(found)}, items {ids}")
        return "\n".join(lines)

    def flagged_bins(self, step=BIN):
        """Bins where an agent other than the object detector asserts a violence concept."""
        if not hasattr(self, "_bins"):
            self.timeline(step)
        return sorted(b for b, slots in self._bins.items()
                      if any(set(found) & VIOLENT for who, found in slots.items() if who != "detector"))

    def unread(self, step=BIN):
        """Flagged bins whose video descriptions the agent has not been shown yet."""
        seen = [self.graph.nodes[i].t_obs for i in self.shown if self.graph.nodes[i].kind == "caption"]
        missing = []
        for b in self.flagged_bins(step):
            lo, hi = b * step, (b + 1) * step
            if not any(a < hi and e > lo for a, e in seen):
                missing.append(f"{lo}-{hi:.0f}s")
        return missing

    def overview_v2(self, variant="c"):
        counts = {}
        for n in self.nodes:
            counts[n.kind] = counts.get(n.kind, 0) + 1
        return (f"Clip length {self.duration:.0f} s. Evidence items: "
                + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
                + f".\nTimeline (concepts asserted per {BIN} s, by agent):\n" + self.timeline()
                + "\nCorroborated events (items from different agents that support each other):\n" + self.events()
                + ("\nVideo descriptions, the whole clip:\n" + self.lines([n for n in self.nodes if n.kind == "caption"], limit=10 ** 6)
                   if variant == "c" else "\nAll evidence items, sorted by time:\n" + self.lines(self.nodes, limit=10 ** 6)))

    def window(self, start, end, kind=None):
        found = [n for n in self.nodes if n.t_obs[0] <= end and n.t_obs[1] >= start
                 and (kind is None or n.kind == kind)]
        return self.lines(found)

    def search(self, concept):
        concept = ALIASES.get(concept, concept)
        if concept not in LEXICON and concept not in ("threat",):
            return f"unknown concept '{concept}'; use one of: " + ", ".join(sorted(set(LEXICON) | {"threat"}))
        return self.lines(n for n in self.nodes if concept in concepts(n)[0])

    def links(self, short_id):
        node_id = self.long.get(short_id)
        if node_id is None:
            return f"no item {short_id}"
        if not self.edges:
            return "links are not available"
        out = []
        for edge, other in self.graph.neighbours(node_id):
            if self.visible(other.id) and edge.type != EdgeType.DERIVED_FROM:
                direction = "->" if edge.src == node_id else "<-"
                score = f" {edge.score:.2f}" if edge.score is not None and self.confidence else ""
                out.append(f"{edge.type.value}{score} {direction} {self.line(other)}")
        return "\n".join(out[:LIST_LIMIT]) or "no links"

    def provenance(self, short_id):
        node_id = self.long.get(short_id)
        if node_id is None:
            return f"no item {short_id}"
        s = self.graph.nodes[node_id].source
        return f"{short_id}: produced by agent '{s.producer}' with {s.version}, input position {s.offset}"


# --- S3: reasoning agent on the graph ----------------------------------------

S3_SYSTEM = """You are the reasoning agent of a video review system. Perception agents have already processed one video clip and stored what they found as evidence items in a shared evidence graph: video descriptions (one per 4 seconds), object detections, sound events, speech transcripts, and speech tagged as threat, distress or report of violence. Items carry a time span and, where available, a confidence. Links between items say whether two items are the same event, support each other, or contradict each other. Detectors and descriptions can be wrong; weigh them against each other.

You read the graph with tools. Reply with exactly one JSON object per turn, nothing else:
{"tool": "window", "start": 10, "end": 30}            items overlapping a time span
{"tool": "window", "start": 10, "end": 30, "kind": "caption"}   same, one kind only (caption, detection, sound_event, transcript_segment, speech_tag)
{"tool": "search", "concept": "gun"}                  items asserting a concept: gun, knife, fire, fight, crash, injury, distress, threat, emergency, crowd, vehicle
{"tool": "links", "id": "n12"}                        items linked to n12 (same_event, supports, contradicts)
{"tool": "provenance", "id": "n12"}                   which agent and model produced n12
{"tool": "done"}                                      you have enough evidence to answer
You have at most %d tool calls."""


def run_s3(llm, view):
    usage, trace = Usage(), []
    messages = [{"role": "system", "content": S3_SYSTEM % MAX_STEPS},
                {"role": "user", "content": "Task: " + QUESTION.split("\n")[0] + "\n\n" + view.overview()}]
    for _ in range(MAX_STEPS):
        text, u, _ = llm.chat(messages, max_tokens=120)
        usage.add(u)
        messages.append({"role": "assistant", "content": text})
        action = parse_action(text)
        trace.append(action or {"invalid": text[:200]})
        if action is None:
            result = "Not a valid tool call. Reply with one JSON object."
        elif action.get("tool") == "done":
            break
        else:
            result = call_tool(view, action)
        messages.append({"role": "user", "content": result})
    return final_turn(llm, messages, usage, view, trace)


S3V2_SYSTEM = """You are the reasoning agent of a video review system. Perception agents have already processed one video clip and stored what they found as evidence items in a shared evidence graph: video descriptions (one per 4 seconds), object detections, sound events, speech transcripts, and speech tagged as threat, distress or report of violence. Items carry a time span and, where available, a confidence. Links say whether items from different agents support each other. Detectors and descriptions can be wrong: an object detector alone is weak evidence, a description or several agents agreeing is stronger.

{intro}

Reply with exactly one JSON object per turn, nothing else:
{"tool": "window", "start": 10, "end": 30}            items overlapping a time span
{"tool": "window", "start": 10, "end": 30, "kind": "caption"}   same, one kind only (caption, detection, sound_event, transcript_segment, speech_tag)
{"tool": "event", "n": 2}                             all items of corroborated event 2
{"tool": "search", "concept": "gun"}                  items asserting a concept: gun, knife, fire, fight, crash, injury, distress, threat, emergency, crowd, vehicle
{"tool": "links", "id": "n12"}                        items linked to n12 (same_event, supports, contradicts)
{"tool": "done"}                                      you have read enough to answer
You have at most %d tool calls."""


S3V2_INTRO = {
    "c": "You start with the video descriptions of the whole clip, a timeline of what each agent found, and the "
         "corroborated events. The object detector is often wrong on films (it mistakes phones, cameras and tools "
         "for guns), so the timeline only shows its findings where another agent agrees. Use the tools to check the "
         "sound, speech and detections behind the flagged spans and events before answering.",
    "d": "You start with every evidence item of the clip, a timeline of what each agent found, and the corroborated "
         "events. The object detector is often wrong on films (it mistakes phones, cameras and tools for guns), so "
         "the timeline only shows its findings where another agent agrees. Use the tools to follow the links of "
         "items you are unsure about before answering.",
}


def run_s3v2(llm, view, variant="c"):
    usage, trace = Usage(), []
    system = S3V2_SYSTEM.replace("{intro}", S3V2_INTRO[variant]) % MAX_STEPS_V2
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Task: " + QUESTION.split("\n")[0] + "\n\n" + view.overview_v2(variant)}]
    for _ in range(MAX_STEPS_V2):
        text, u, _ = llm.chat(messages, max_tokens=120)
        usage.add(u)
        messages.append({"role": "assistant", "content": text})
        action = parse_action(text)
        trace.append(action or {"invalid": text[:200]})
        if action is None:
            result = "Not a valid tool call. Reply with one JSON object."
        elif action.get("tool") == "done":
            missing = view.unread()
            if not missing:
                break
            result = ("Not yet: read the video descriptions of these flagged spans first: "
                      + ", ".join(missing[:8]) + ". Use window with kind caption.")
        elif action.get("tool") == "event":
            groups = view.event_groups()
            try:
                members = groups[int(action["n"]) - 1][1]
                result = view.lines(sorted(members, key=lambda m: m.t_obs[0]))
            except (KeyError, ValueError, IndexError, TypeError):
                result = f"no event {action.get('n')}"
        else:
            result = call_tool(view, action)
        messages.append({"role": "user", "content": result})
    return final_turn(llm, messages, usage, view, trace)


def parse_action(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        action = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return action if isinstance(action, dict) and "tool" in action else None


def call_tool(view, a):
    try:
        tool = a["tool"]
        if tool == "window":
            return view.window(float(a["start"]), float(a["end"]), a.get("kind"))
        if tool == "search":
            return view.search(str(a["concept"]).lower())
        if tool == "links":
            return view.links(str(a["id"]))
        if tool == "provenance":
            return view.provenance(str(a["id"]))
    except (KeyError, TypeError, ValueError) as e:
        return f"bad arguments: {e}"
    return f"unknown tool {a.get('tool')}"


def final_turn(llm, messages, usage, view, trace):
    messages = messages + [{"role": "user", "content": QUESTION + "\n" + CITE_LINE}]
    text, u, logprobs = llm.chat(messages, max_tokens=120, logprobs=True)
    usage.add(u)
    decision, segments = parse_answer("\n".join(text.strip().splitlines()[:2]))
    cited_short = re.findall(r"\bn\d+\b", "\n".join(text.strip().splitlines()[2:]))
    cited = [view.long[c] for c in cited_short if c in view.long] if view else []
    score = yes_probability(logprobs)
    if score is None:
        score = 1.0 if decision == "yes" else 0.0
    return {"score": round(score, 4), "decision": decision, "segments": segments, "cited": cited,
            "answer": text.strip(), "usage": usage.as_dict(), "trace": trace}


# --- S4: flat text --------------------------------------------------------

S4_SYSTEM = ("You are the reasoning agent of a video review system. Perception agents have processed one video "
             "clip; below is everything they found, sorted by time: video descriptions (one per 4 seconds), object "
             "detections, sound events, speech transcripts, and speech tagged as threat, distress or report of "
             "violence. Detectors and descriptions can be wrong; weigh them against each other.")


def run_s4(llm, view):
    usage = Usage()
    messages = [{"role": "system", "content": S4_SYSTEM},
                {"role": "user", "content": f"Clip length {view.duration:.0f} s.\n" + view.lines(view.nodes, limit=10 ** 6)}]
    return final_turn(llm, messages, usage, view, [])


def run_s4g(llm, view):
    """Diagnostic: S4's flat text plus the graph summaries (timeline, events), one pass, no tools."""
    usage = Usage()
    messages = [{"role": "system", "content": S4_SYSTEM + " A timeline of what each agent found and the "
                 "corroborated events (items from different agents that support each other) come first."},
                {"role": "user", "content": view.overview_v2("d")}]
    return final_turn(llm, messages, usage, view, [])


# --- S2: message passing ---------------------------------------------------

AGENT_ROLE = {
    "video": "the video agent: you hold descriptions of the clip every 4 seconds and object detections",
    "audio_events": "the sound agent: you hold the sound events an audio tagger found",
    "speech": "the speech agent: you hold the transcript and the speech tagged as threat, distress or report",
}
REPORT_ASK = ("Write a short report (at most 120 words) for the reasoning agent, who must decide whether the clip "
              "needs a security operator's attention. Say what you found and when (in seconds), and how sure you "
              "are. Do not invent anything that is not in your data.")
S2_SYSTEM = """You are the reasoning agent of a video review system. Three perception agents processed one video clip and each sent you a report: video, sound and speech. Their tools can be wrong. You may ask an agent a follow-up question. Reply with exactly one JSON object per turn, nothing else:
{"ask": "video", "question": "..."}      ask an agent (video, audio_events or speech)
{"done": true}                           you have enough to answer
You may ask at most %d questions."""


def agent_view(view, name):
    producers = AGENTS[name]
    nodes = [n for n in view.nodes if n.source.producer in producers]
    return f"Clip length {view.duration:.0f} s. Your data:\n" + view.lines(nodes, limit=10 ** 6)


def run_s2(llm, view):
    usage, trace, chats = Usage(), [], {}
    reports = []
    for name in AGENTS:
        chats[name] = [{"role": "system", "content": f"You are {AGENT_ROLE[name]}.\n" + agent_view(view, name)},
                       {"role": "user", "content": REPORT_ASK}]
        text, u, _ = llm.chat(chats[name], max_tokens=200)
        usage.add(u)
        chats[name].append({"role": "assistant", "content": text})
        reports.append(f"Report from the {name} agent:\n{text.strip()}")
    messages = [{"role": "system", "content": S2_SYSTEM % MAX_QUESTIONS},
                {"role": "user", "content": "Task: " + QUESTION.split("\n")[0] + "\n\n" + "\n\n".join(reports)}]
    for _ in range(MAX_QUESTIONS):
        text, u, _ = llm.chat(messages, max_tokens=120)
        usage.add(u)
        messages.append({"role": "assistant", "content": text})
        m = re.search(r"\{.*\}", text, re.S)
        try:
            action = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            action = None
        trace.append(action or {"invalid": text[:200]})
        if not isinstance(action, dict) or action.get("done") or action.get("ask") not in chats:
            if isinstance(action, dict) and action.get("done"):
                break
            messages.append({"role": "user", "content": "Not a valid message. Reply with one JSON object."})
            continue
        name = action["ask"]
        chats[name].append({"role": "user", "content": str(action.get("question", ""))[:500]})
        answer, u, _ = llm.chat(chats[name], max_tokens=200)
        usage.add(u)
        chats[name].append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": f"Answer from the {name} agent:\n{answer.strip()}"})
    # S2 shares no item ids, so its citations are empty by construction.
    out = final_turn(llm, messages, usage, None, trace)
    out["cited"] = []
    return out


# --- running ----------------------------------------------------------------

SYSTEMS = {"s2": run_s2, "s3": run_s3, "s3v2": run_s3v2, "s4": run_s4, "s4g": run_s4g}


def run_clip(args, llm, cid, earlier):
    graph = clip_graph(args.graphs, cid)
    delete = []
    if earlier is not None:
        cited = earlier.get(cid, {}).get("cited", [])
        if args.delete_cited:
            delete = cited
        elif args.delete_random:
            pool = sorted(set(graph.nodes) - set(cited))
            delete = random.Random(f"{args.seed}:{cid}").sample(pool, min(len(cited), len(pool)))
    view = View(graph, drop=args.drop, delete=delete, edges=not args.no_edges, confidence=not args.no_confidence)
    start = time.time()
    try:
        out = SYSTEMS[args.system](llm, view, args.variant) if args.system == "s3v2" else SYSTEMS[args.system](llm, view)
    except RuntimeError as e:  # recorded as a failure: no decision, uninformative score
        out = {"score": 0.5, "decision": None, "segments": [], "cited": [], "answer": "", "error": str(e),
               "usage": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}, "trace": []}
    out.update(clip_id=cid, deleted=len(delete), nodes_visible=len(view.nodes), seconds=round(time.time() - start, 1))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--graphs", required=True, help="graphs root (one folder per agent, no clips.csv)")
    parser.add_argument("--out", required=True, help="JSONL file, one line per clip; existing clips are skipped")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--no-think", action="store_true", help="turn off a thinking mode (Qwen3.6)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--variant", choices=S3V2_VARIANTS, default="c", help="S3 v2 variant (dev-selected: c)")
    parser.add_argument("--clips", help="file with clip ids to run (default: all in the graphs root)")
    parser.add_argument("--drop", nargs="*", default=[], choices=list(AGENTS))
    parser.add_argument("--no-edges", action="store_true")
    parser.add_argument("--no-confidence", action="store_true")
    parser.add_argument("--delete-cited", help="earlier run (JSONL): hide the nodes it cited")
    parser.add_argument("--delete-random", help="earlier run (JSONL): hide as many random uncited nodes")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    graphs = Path(args.graphs)
    assert not (graphs / "clips.csv").exists(), "clips.csv maps clip ids to files; keep it away from the agent"
    earlier_path = args.delete_cited or args.delete_random
    earlier = None
    if earlier_path:
        earlier = {r["clip_id"]: r for r in map(json.loads, open(earlier_path))}

    cids = sorted(p.stem for p in (graphs / "video").glob("x*.jsonl"))
    if args.clips:
        cids = [c for c in cids if c in set(Path(args.clips).read_text().split())]
    out = Path(args.out)
    done = {json.loads(l)["clip_id"] for l in open(out)} if out.exists() else set()
    todo = [c for c in cids if c not in done][:args.limit]
    llm = LLM(args.server, args.model, no_think=args.no_think)
    config = {"system": args.system, "s3v2": args.variant if args.system == "s3v2" else None, "model": args.model, "no_think": args.no_think, "task": TASK_VERSION, "drop": args.drop,
              "no_edges": args.no_edges, "no_confidence": args.no_confidence,
              "delete": "cited" if args.delete_cited else "random" if args.delete_random else None,
              "from": earlier_path}
    t0 = time.time()
    with open(out, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        for k, result in enumerate(pool.map(lambda c: run_clip(args, llm, c, earlier), todo), 1):
            f.write(json.dumps({**result, "config": config}, ensure_ascii=False) + "\n")
            f.flush()
            if k % 25 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
