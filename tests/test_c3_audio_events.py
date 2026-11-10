import numpy as np

import audio_events as ae
import audio_precheck as ap
from evgraph import EvidenceGraph, Modality

LABELS = {0: "Speech", 1: "Gunshot, gunfire", 2: "Machine gun", 3: "Explosion"}


def test_runs_finds_consecutive_windows():
    assert ae.runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]
    assert ae.runs(np.array([], bool)) == []


def test_event_nodes_merge_windows_and_keep_peak_score():
    starts = np.array([0.0, 5.0, 10.0, 15.0])
    scores = np.zeros((4, 4), np.float32)
    scores[1, 1], scores[2, 2] = 0.4, 0.8      # gunshot group over windows 1-2, peak on machine gun
    scores[0, 0] = 0.9                         # speech is not an event group
    nodes = ae.event_nodes("xabc", starts, scores, LABELS, duration=22.0)
    assert len(nodes) == 1
    node = nodes[0]
    assert node.modality == Modality.AUDIO_EVENT
    assert node.content["event"] == "gunshot" and node.content["top_class"] == "Machine gun"
    assert node.confidence == 0.8 and node.t_obs == (5.0, 20.0)
    assert node.t_rec == node.t_obs[1]


def test_end_is_clipped_to_duration_and_graph_round_trips(tmp_path):
    starts = np.array([0.0])
    scores = np.zeros((1, 4), np.float32)
    scores[0, 3] = 0.5
    nodes = ae.event_nodes("xabc", starts, scores, LABELS, duration=7.0)
    assert nodes[0].t_obs == (0.0, 7.0)
    g = EvidenceGraph(tmp_path / "g.jsonl")
    g.add(nodes[0])
    assert EvidenceGraph(tmp_path / "g.jsonl").in_window(6, 8, Modality.AUDIO_EVENT)[0].id == nodes[0].id


def test_clip_id_hides_the_filename():
    stem = "Film__#00-01-00_00-02-00_label_B2-0-0"
    cid = ae.clip_id(stem)
    assert cid == ae.clip_id(stem) and "label" not in cid and len(cid) == 11
    assert set(ap.EVENT_GROUPS) <= set(ap.GROUPS)


def test_speech_and_audio_agents_use_the_same_clip_ids():
    import speech_asr
    stem = "Film__#00-01-00_00-02-00_label_B2-0-0"
    assert speech_asr.clip_id(stem) == ae.clip_id(stem)


def test_clip_graph_merges_agents_and_cross_agent_edges(tmp_path):
    from evgraph import Edge, EdgeType, Source
    starts = np.array([0.0])
    scores = np.zeros((1, 4), np.float32)
    scores[0, 3] = 0.5
    audio = ae.event_nodes("xabc", starts, scores, LABELS, duration=7.0)[0]
    other = ae.Node(id="xabc:speech:0", modality=Modality.SPEECH, kind="transcript_segment",
                    content={"text": "help"}, t_obs=(1.0, 2.0), t_rec=2.0,
                    source=Source("speech", "0"), confidence=0.9)
    (tmp_path / "edges").mkdir()
    (tmp_path / "audio_events").mkdir()
    (tmp_path / "speech").mkdir()
    EvidenceGraph(tmp_path / "audio_events/xabc.jsonl").add(audio)
    EvidenceGraph(tmp_path / "speech/xabc.jsonl").add(other)
    edges = EvidenceGraph()
    edges.add(audio), edges.add(other)
    edges.path = tmp_path / "edges/xabc.jsonl"
    edges.add(Edge(audio.id, other.id, EdgeType.SAME_EVENT, Source("rules", "0"), 1.0, score=0.7))
    merged = ae.clip_graph(tmp_path, "xabc")
    assert set(merged.nodes) == {audio.id, other.id}
    assert len(merged.neighbours(audio.id)) == 1


def test_speech_nodes_drop_failed_segments():
    import speech
    t = {"clip_id": "xabc", "language": "en", "model": "fw large-v3", "segments": [
        {"start": 0.0, "end": 2.0, "text": "Get down!", "avg_logprob": -0.2, "no_speech_prob": 0.1,
         "compression_ratio": 1.2},
        {"start": 2.0, "end": 4.0, "text": "la la la la", "avg_logprob": -0.3, "no_speech_prob": 0.1,
         "compression_ratio": 3.0},
        {"start": 4.0, "end": 6.0, "text": "uh", "avg_logprob": -1.5, "no_speech_prob": 0.9,
         "compression_ratio": 1.0}]}
    nodes = speech.speech_nodes(t)
    assert [n.content["text"] for n in nodes] == ["Get down!"]
    assert 0.7 < nodes[0].confidence < 0.75 and nodes[0].modality == Modality.SPEECH
