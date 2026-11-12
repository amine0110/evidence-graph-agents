"""Edge builder: link the nodes the perception agents wrote for each clip.

Every node is first mapped to the concepts it asserts (gun, fire, fight, ...)
and, for captions, the concepts it explicitly denies ("No weapons are
visible"). Captions are split into sentences; a sentence with a negation word
denies its concepts instead of asserting them.

Edges, only between nodes of different agents or kinds whose times overlap
(with SLACK seconds of tolerance):
  same_event    both nodes assert some concept; score = overlap of the two
                intervals divided by the shorter one.
  supports      the concepts agree (equal, or a pair in COMPATIBLE); score =
                the lower of the two confidences. Directed from the node
                recorded first to the other.
  contradicts   a caption denies a concept in CONTESTED that the other node
                asserts; score = the asserting node's confidence. Directed
                from the caption. Vehicles, crowds and the like are left out:
                a caption missing a parked car says nothing about violence.
  derived_from  a speech tag to the transcript segment it was read from.
precedes is not written: the order of nodes follows from their times.

Writes <graphs>/edges/<clip>.jsonl. These are rules, not a model: every edge
can be traced to a lexicon entry or a time overlap.
"""

import argparse
import csv
import json
import re
from pathlib import Path

from evgraph import Edge, EdgeType, EvidenceGraph, Source
from evgraph.schema import to_record
from audio_events import clip_graph

PRODUCER = "edges"
VERSION = "edge-rules-v1"
SLACK = 1.0

SOUND = {"gunshot": "gun", "explosion": "fire", "scream": "distress", "crash": "crash",
         "impact": "fight", "siren": "emergency", "crowd": "crowd"}
DETECTION = {"pistol_hand_gun": "gun", "machine_gun": "gun", "RPG": "gun", "knife": "knife",
             "grenade": "fire", "fire": "fire", "smoke": "fire", "police officer": "emergency",
             "car": "vehicle", "motorcycle": "vehicle", "stick": "fight"}
SPEECH = {"threat": "threat", "distress": "distress"}
LEXICON = {
    "gun": r"guns?|handguns?|pistols?|rifles?|firearms?|shoot\w*|shots?|gunfire|gunshots?|weapons?|armed",
    "knife": r"knife|knives|blades?|stab\w*",
    "fire": r"fire|flames?|explosions?|explod\w*|burning|blaze|smoke",
    "fight": r"punch\w*|kick\w*|fight\w*|hit|hits|hitting|struggl\w*|attack\w*|beat|beats|beating|brawl\w*|assault\w*|strikes?|striking|clubs?|bats?|batons?|swings? at|shov\w*|grab\w* \w+ by",
    "crash": r"crash\w*|collid\w*|collisions?|overturn\w*|flip|flips|flipped",
    "injury": r"injur\w*|blood\w*|wound\w*|bleed\w*|motionless|unconscious|lifeless|dead|bod(y|ies) (lies|lie|lying)",
    "distress": r"scream\w*|cry|cries|crying|panic\w*|flee\w*",
    "emergency": r"police|ambulances?|firefighters?|sirens?",
    "crowd": r"crowds?|protest\w*|riot\w*",
    "vehicle": r"cars?|trucks?|vehicles?|motorcycles?|buses|bus",
}
LEXICON = {k: re.compile(rf"\b({v})\b", re.I) for k, v in LEXICON.items()}
# sport and photography senses that the lexicon would otherwise read as violence
NOT_VIOLENT = re.compile(r"\b(shoot\w*|shots?) (the |a |for )?(ball|basket|hoop|goal|puck|photos?|pictures?|videos?|scenes?)\b"
                         r"|\b(basketball|football|soccer|hockey|tennis|baseball|golf)\b[^.]*\b(shoot\w*|shots?|hits?|strikes?|bats?)\b", re.I)
NEGATION = re.compile(r"\b(no|not|without|none|neither|nor)\b", re.I)
CONTESTED = {"gun", "knife", "fire", "fight", "injury", "crash", "distress"}
COMPATIBLE = {frozenset(p) for p in [("threat", "gun"), ("threat", "knife"), ("threat", "fight"),
                                     ("distress", "fight"), ("distress", "injury"), ("distress", "gun"),
                                     ("distress", "fire"), ("distress", "crash"), ("injury", "fight"),
                                     ("injury", "gun"), ("injury", "crash"), ("injury", "fire"),
                                     ("crash", "vehicle")]}


def text_concepts(text):
    """(asserted, denied) concepts of a text, sentence by sentence."""
    asserted, denied = set(), set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        found = {c for c, rx in LEXICON.items() if rx.search(NOT_VIOLENT.sub(" ", sentence))}
        (denied if NEGATION.search(sentence) else asserted).update(found)
    return asserted, denied


def concepts(node):
    """(asserted, denied) concepts of one node."""
    c = node.content
    if node.kind == "sound_event":
        return {SOUND[c["event"]]} if c["event"] in SOUND else set(), set()
    if node.kind == "detection":
        return {DETECTION[c["object"]]} if c["object"] in DETECTION else set(), set()
    if node.kind == "caption":
        return text_concepts(c["text"])
    if node.kind == "speech_tag":
        if c["tag"] in SPEECH:
            return {SPEECH[c["tag"]]}, set()
        return text_concepts(c["text"])[0], set()  # a report asserts what it describes
    return set(), set()


def overlap(a, b):
    lo, hi = max(a.t_obs[0], b.t_obs[0]), min(a.t_obs[1], b.t_obs[1])
    if hi + SLACK < lo:
        return None
    shorter = max(min(a.t_obs[1] - a.t_obs[0], b.t_obs[1] - b.t_obs[0]), SLACK)
    return min(max(hi - lo, 0.0) + SLACK, shorter) / shorter


def agrees(x, y):
    return bool(x & y) or any(frozenset((p, q)) in COMPATIBLE for p in x for q in y)


def clip_edges(graph, cid):
    source = Source(producer=PRODUCER, version=VERSION, input_ref=cid)
    nodes = sorted(graph.nodes.values(), key=lambda n: (n.t_obs[0], n.id))
    tagged = [(n, *concepts(n)) for n in nodes]
    tagged = [t for t in tagged if t[1] or t[2]]
    edges = []
    for i, (a, a_pos, a_neg) in enumerate(tagged):
        for b, b_pos, b_neg in tagged[i + 1:]:
            if b.t_obs[0] > a.t_obs[1] + SLACK:
                break
            if (a.source.producer, a.kind) == (b.source.producer, b.kind):
                continue
            ov = overlap(a, b)
            if ov is None:
                continue
            if a_pos and b_pos:
                edges.append(Edge(a.id, b.id, EdgeType.SAME_EVENT, source, 1.0, round(ov, 4)))
                if agrees(a_pos, b_pos):
                    first, second = (a, b) if a.t_rec <= b.t_rec else (b, a)
                    s = round(min(a.confidence, b.confidence), 4)
                    edges.append(Edge(first.id, second.id, EdgeType.SUPPORTS, source, s, s))
            for cap, cap_neg, other, other_pos in ((a, a_neg, b, b_pos), (b, b_neg, a, a_pos)):
                if cap.kind == "caption" and cap_neg & other_pos & CONTESTED:
                    s = round(other.confidence, 4)
                    edges.append(Edge(cap.id, other.id, EdgeType.CONTRADICTS, source, s, s))
    for n in nodes:
        if n.kind == "speech_tag" and n.content.get("from") in graph.nodes:
            edges.append(Edge(n.id, n.content["from"], EdgeType.DERIVED_FROM, source, 1.0))
    return edges


def build(graphs_root):
    graphs_root = Path(graphs_root)
    out_dir = graphs_root / PRODUCER
    out_dir.mkdir(parents=True, exist_ok=True)
    cids = [r["clip_id"] for r in csv.DictReader(open(graphs_root / "clips.csv"))]
    counts = {t.value: 0 for t in EdgeType}
    for cid in cids:
        target = out_dir / f"{cid}.jsonl"
        target.unlink(missing_ok=True)  # rebuild from the agents' nodes only
        graph = clip_graph(graphs_root, cid)
        edges = clip_edges(graph, cid)
        lines = []
        for edge in edges:
            graph.add(edge)  # validates both ends
            lines.append(to_record(edge))
            counts[edge.type.value] += 1
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines))
    return {"clips": len(cids), "edges": counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", required=True, help="graphs root with one folder per agent")
    args = parser.parse_args(argv)
    print(build(args.graphs))


if __name__ == "__main__":
    main()
