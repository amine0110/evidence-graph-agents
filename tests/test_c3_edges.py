from evgraph import EdgeType, EvidenceGraph, Modality, Node, Source

from edges import clip_edges, concepts, text_concepts
from speech_tags import tag_nodes


def node(id, kind, content, t, producer, conf=0.8, modality=Modality.VIDEO):
    return Node(id=id, modality=modality, kind=kind, content=content, t_obs=t, t_rec=t[1],
                source=Source(producer=producer, version="test"), confidence=conf)


def test_negated_sentences_deny_instead_of_assert():
    pos, neg = text_concepts("A man fires a pistol. No injuries or fire are visible.")
    assert pos == {"gun"}
    assert neg == {"injury", "fire"}


def test_caption_contradicts_detector_and_sound_supports():
    g = EvidenceGraph()
    cap = g.add(node("c", "caption", {"text": "Two men talk. No weapons are visible."}, (0, 4), "video"))
    gun = g.add(node("d", "detection", {"object": "pistol_hand_gun"}, (1, 2), "video", 0.6))
    car = g.add(node("v", "detection", {"object": "car"}, (1, 3), "video", 0.9))
    shot = g.add(node("s", "sound_event", {"event": "gunshot"}, (0, 10), "audio_events", 0.7,
                      Modality.AUDIO_EVENT))
    edges = clip_edges(g, "x")
    kinds = {(e.src, e.dst, e.type) for e in edges}
    assert (cap.id, gun.id, EdgeType.CONTRADICTS) in kinds
    assert (cap.id, shot.id, EdgeType.CONTRADICTS) in kinds
    assert not any(e.type == EdgeType.CONTRADICTS and e.dst == car.id for e in edges)  # vehicles not contested
    assert any(e.type == EdgeType.SUPPORTS and {e.src, e.dst} == {gun.id, shot.id} for e in edges)
    support = next(e for e in edges if e.type == EdgeType.SUPPORTS)
    assert support.score == 0.6  # lower confidence of the two


def test_no_edges_between_far_apart_nodes():
    g = EvidenceGraph()
    g.add(node("a", "detection", {"object": "knife"}, (0, 1), "video"))
    g.add(node("b", "sound_event", {"event": "scream"}, (20, 30), "audio_events", modality=Modality.AUDIO_EVENT))
    assert clip_edges(g, "x") == []


def test_speech_tags_keep_confident_non_none_labels_and_link_back():
    record = {"clip_id": "x1", "model": "m", "segments": [
        {"node": "x1:speech:0", "start": 0.0, "end": 2.0, "text": "Drop it or I shoot!",
         "probs": {"threat": 0.9, "distress": 0.05, "report": 0.03, "none": 0.02}},
        {"node": "x1:speech:1", "start": 2.0, "end": 4.0, "text": "Nice weather.",
         "probs": {"threat": 0.01, "distress": 0.01, "report": 0.01, "none": 0.97}},
        {"node": "x1:speech:2", "start": 4.0, "end": 5.0, "text": "Hmm",
         "probs": {"threat": 0.4, "distress": 0.3, "report": 0.0, "none": 0.3}}]}
    nodes = tag_nodes(record)
    assert [n.id for n in nodes] == ["x1:speech:0:tag"]
    assert nodes[0].confidence == 0.9 and concepts(nodes[0])[0] == {"threat"}
    g = EvidenceGraph()
    g.add(Node(id="x1:speech:0", modality=Modality.SPEECH, kind="transcript_segment",
               content={"text": "Drop it or I shoot!"}, t_obs=(0.0, 2.0), t_rec=2.0,
               source=Source(producer="speech", version="t"), confidence=0.9))
    g.add(nodes[0])
    assert any(e.type == EdgeType.DERIVED_FROM and e.dst == "x1:speech:0" for e in clip_edges(g, "x1"))


def test_sport_senses_are_not_violence():
    assert text_concepts("A player in green shoots the ball toward the basket.") == (set(), set())
    assert text_concepts("In the basketball game a player takes a long shot.")[0] == set()
    assert text_concepts("A man shoots another man with a pistol.")[0] == {"gun"}


def test_club_and_motionless_body():
    pos, _ = text_concepts("A man raises a club to strike a person lying motionless in the grass.")
    assert {"fight", "injury"} <= pos
