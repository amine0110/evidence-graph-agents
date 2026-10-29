import pytest

from evgraph import Edge, EdgeType, EvidenceGraph, Modality, Node, Source, Veracity

SRC = Source(producer="test", version="0")


def headline(node_id, day, label=None):
    return Node(id=node_id, modality=Modality.TEXT, kind="headline",
                content={"title": node_id}, t_obs=(day, day), t_rec=day,
                source=SRC, confidence=1.0, label=label)


def test_graded_edges_need_scores_in_range():
    with pytest.raises(ValueError):
        Edge("a", "b", EdgeType.SUPPORTS, SRC, confidence=0.9)
    with pytest.raises(ValueError):
        Edge("a", "b", EdgeType.CONTRADICTS, SRC, confidence=0.9, score=1.5)
    with pytest.raises(ValueError):
        Edge("a", "b", EdgeType.PRECEDES, SRC, confidence=1.0, score=0.3)
    with pytest.raises(ValueError):
        Edge("a", "a", EdgeType.SAME_EVENT, SRC, confidence=1.0, score=0.8)


def test_nodes_are_immutable_and_edges_need_known_nodes():
    g = EvidenceGraph()
    g.add(headline("a", 1))
    with pytest.raises(ValueError):
        g.add(headline("a", 2))
    with pytest.raises(ValueError):
        g.add(Edge("a", "zzz", EdgeType.SUPPORTS, SRC, confidence=1.0, score=0.5))


def test_past_only_and_typed_neighbours():
    g = EvidenceGraph()
    for node_id, day in [("old", 1), ("query", 5), ("future", 9)]:
        g.add(headline(node_id, day))
    g.add(Edge("query", "old", EdgeType.CONTRADICTS, SRC, confidence=0.8, score=0.7))
    g.add(Edge("query", "future", EdgeType.SUPPORTS, SRC, confidence=0.8, score=0.9))
    g.add(Edge("old", "query", EdgeType.SAME_EVENT, SRC, confidence=1.0, score=0.6))

    everything = g.neighbours("query")
    assert {n.id for _, n in everything} == {"old", "future"}
    past = g.neighbours("query", recorded_before=5)
    assert {n.id for _, n in past} == {"old"}
    contradicting = g.neighbours("query", types=[EdgeType.CONTRADICTS])
    assert [(e.type, n.id) for e, n in contradicting] == [(EdgeType.CONTRADICTS, "old")]
    # same_event is symmetric, so it is found from either end.
    same = g.neighbours("query", types=[EdgeType.SAME_EVENT])
    assert [n.id for _, n in same] == ["old"]


def test_time_window_and_veracity_history(tmp_path):
    path = tmp_path / "graph.jsonl"
    g = EvidenceGraph(path)
    g.add(Node(id="shot", modality=Modality.AUDIO_EVENT, kind="sound_event",
               content={"class": "gunshot"}, t_obs=(12.0, 12.4), t_rec=0.0,
               source=Source("panns", "cnn14", "clip.mp4", "12.0-12.4s"), confidence=0.83))
    g.add(Node(id="gun", modality=Modality.VIDEO, kind="detection",
               content={"class": "weapon"}, t_obs=(11.5, 13.0), t_rec=0.0,
               source=Source("yolo", "v8-weapons"), confidence=0.71))
    g.add(Veracity("gun", 0.71, at=1.0, rule="init"))
    g.add(Veracity("gun", 0.9, at=2.0, rule="graded", evidence=("shot->gun",)))

    assert [n.id for n in g.in_window(12.0, 12.2)] == ["gun", "shot"]
    assert [n.id for n in g.in_window(0, 20, Modality.AUDIO_EVENT)] == ["shot"]

    reloaded = EvidenceGraph(path)
    assert reloaded.veracity("gun").value == 0.9
    assert len(reloaded.beliefs["gun"]) == 2
    assert reloaded.nodes["shot"].source.offset == "12.0-12.4s"
