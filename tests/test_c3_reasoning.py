import math

from evgraph import Edge, EdgeType, EvidenceGraph, Modality, Node, Source

import reasoning
from task import parse_answer


def graph():
    g = EvidenceGraph()
    src = lambda p: Source(producer=p, version="t")
    g.add(Node("x:video:caption:0", Modality.VIDEO, "caption", {"text": "A man fires a pistol."}, (0, 4), 4,
               src("video"), 1.0))
    g.add(Node("x:audio_events:0", Modality.AUDIO_EVENT, "sound_event",
               {"event": "gunshot", "top_class": "Gunshot, gunfire", "peak_score": 0.7, "peak_time": 2},
               (0, 10), 10, src("audio_events"), 0.7))
    g.add(Node("x:speech:0", Modality.SPEECH, "transcript_segment", {"text": "Get down!", "language": "en"},
               (1, 2), 2, src("speech"), 0.9))
    g.add(Edge("x:audio_events:0", "x:video:caption:0", EdgeType.SUPPORTS, src("edges"), 0.7, 0.7))
    return g


class FakeLLM:
    """Plays back scripted replies; the last reply carries first-token log probabilities."""

    def __init__(self, replies, p_yes=0.8):
        self.replies, self.p_yes, self.seen = list(replies), p_yes, []

    def chat(self, messages, max_tokens=None, logprobs=False):
        self.seen.append(list(messages))
        lp = {"content": [{"top_logprobs": [{"token": "Yes", "logprob": math.log(self.p_yes)},
                                            {"token": "No", "logprob": math.log(1 - self.p_yes)}]}]}
        return self.replies.pop(0), {"prompt_tokens": 10, "completion_tokens": 2}, lp if logprobs else None


def test_parse_answer():
    assert parse_answer("Yes\n12-20, 31.5-40") == ("yes", [(12.0, 20.0), (31.5, 40.0)])
    assert parse_answer("No\nnone") == ("no", [])


def test_s3_uses_tools_and_maps_citations_back():
    view = reasoning.View(graph())
    llm = FakeLLM(['{"tool": "search", "concept": "gun"}', '{"tool": "links", "id": "n1"}',
                   '{"tool": "done"}', "Yes\n0-10\nn1, n2, n99"])
    out = reasoning.run_s3(llm, view)
    assert out["decision"] == "yes" and out["segments"] == [(0.0, 10.0)]
    assert abs(out["score"] - 0.8) < 1e-6
    assert out["cited"] == ["x:video:caption:0", "x:audio_events:0"]  # n99 does not exist
    tool_results = [m[-1]["content"] for m in llm.seen[1:3]]
    assert "gunshot" in tool_results[0] and "supports" in tool_results[1]
    assert out["usage"]["calls"] == 4


def test_ablations_hide_agents_edges_and_confidence():
    view = reasoning.View(graph(), drop=["audio_events"], edges=False, confidence=False)
    assert all(n.source.producer != "audio_events" for n in view.nodes)
    assert view.links("n1") == "links are not available"
    assert "conf" not in view.lines(view.nodes)


def test_s4_and_s2_end_with_the_same_question():
    out4 = reasoning.run_s4(FakeLLM(["No\nnone\nnone"], p_yes=0.1), reasoning.View(graph()))
    assert out4["decision"] == "no" and abs(out4["score"] - 0.1) < 1e-6
    llm = FakeLLM(["video report", "sound report", "speech report", '{"ask": "speech", "question": "who?"}',
                   "a man", '{"done": true}', "Yes\n0-4\nn1"])
    out2 = reasoning.run_s2(llm, reasoning.View(graph()))
    assert out2["decision"] == "yes" and out2["cited"] == []
    assert llm.seen[-1][-1]["content"].startswith(reasoning.QUESTION)


def test_search_maps_synonyms_and_lists_concepts():
    view = reasoning.View(graph())
    assert "gunshot" in view.search("shooting")
    assert view.search("kittens").startswith("unknown concept")


def test_s3v2_timeline_and_events():
    view = reasoning.View(graph())
    tl = view.timeline()
    assert "0-10s" in tl and "sound: gun 0.70" in tl and "video: gun 1.00" in tl
    ev = view.events()
    assert ev.startswith("event 1:") and "sound" in ev and "video" in ev
    llm = FakeLLM(['{"tool": "event", "n": 1}', '{"tool": "done"}', "Yes\n0-10\nn1"])
    out = reasoning.run_s3v2(llm, view)
    assert out["decision"] == "yes" and "gunshot" in llm.seen[1][-1]["content"]
    assert reasoning.View(graph(), edges=False).events() == "links are not available"


def test_paraphrase_changes_only_the_opening_paragraph():
    assert reasoning.S3_INTRO in reasoning.S3_SYSTEM
    p2 = reasoning.S3_SYSTEM.replace(reasoning.S3_INTRO, reasoning.S3_INTRO_P2)
    assert p2 != reasoning.S3_SYSTEM and p2.split("\n\n")[1:] == reasoning.S3_SYSTEM.split("\n\n")[1:]
    old = reasoning.PARAPHRASE
    reasoning.PARAPHRASE = True
    llm = FakeLLM(["No\nnone\nnone"])
    reasoning.run_s4(llm, reasoning.View(graph()))
    assert llm.seen[0][0]["content"] == reasoning.S4_SYSTEM_P2
    reasoning.PARAPHRASE = old
