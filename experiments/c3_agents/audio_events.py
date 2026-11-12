"""Audio-event perception agent for C3 Domain A.

Turns the AudioSet tagger scores of a clip (10 s windows every 5 s, computed
by `audio_precheck.py scan`) into sound_event nodes of the evidence graph:
one node per run of consecutive windows in which an event group (gunshot,
explosion, scream, crash, crowd, siren, impact) scores at least NODE_MIN.
The node keeps the tagger's peak score as its confidence, so the reasoning
agent sees how sure the tagger was rather than a yes/no.

Each agent writes its own graph file per clip, <graphs>/<agent>/<clip id>.jsonl,
so rerunning one agent never touches the others; clip_graph() merges them.
Graphs are named by a neutral clip id, never by the XD-Violence filename,
which contains the labels; clips.csv maps one to the other and must not be
given to the reasoning agent.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import audio_precheck as ap
from evgraph import Edge, EvidenceGraph, Modality, Node, Source, Veracity

NODE_MIN = 0.1
PRODUCER = "audio_events"
VERSION = f"{ap.MODEL}@win{ap.WIN:g}s-hop{ap.HOP:g}s"


def clip_id(stem):
    """Neutral, stable id for a clip: does not reveal its label."""
    return "x" + hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]


def runs(mask):
    """(first, last) index of each run of True values."""
    out, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        if not on and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def event_nodes(cid, starts, scores, labels, duration):
    """sound_event nodes for one clip from its window scores (windows x classes)."""
    index = {name: i for i, name in labels.items()}
    nodes = []
    for group in ap.EVENT_GROUPS:
        columns = [index[n] for n in ap.GROUPS[group] if n in index]
        if not len(scores) or not columns:
            continue
        group_scores = scores[:, columns]
        best = group_scores.max(axis=1)
        for first, last in runs(best >= NODE_MIN):
            peak = first + int(np.argmax(best[first:last + 1]))
            start = float(starts[first])
            end = float(min(starts[last] + ap.WIN, duration))
            top = labels[columns[int(np.argmax(group_scores[peak]))]]
            nodes.append(Node(
                id=f"{cid}:audio:{group}:{len(nodes)}",
                modality=Modality.AUDIO_EVENT,
                kind="sound_event",
                content={"event": group, "top_class": top, "peak_score": round(float(best[peak]), 4),
                         "peak_time": round(float(starts[peak] + ap.WIN / 2), 2)},
                t_obs=(round(start, 2), round(end, 2)),
                t_rec=round(end, 2),
                source=Source(producer=PRODUCER, version=VERSION, input_ref=cid,
                              offset=f"{start:.1f}-{end:.1f}s"),
                confidence=round(float(best[peak]), 4),
            ))
    return sorted(nodes, key=lambda n: (n.t_obs[0], n.id))


def clip_graph(graphs_root, cid):
    """In-memory graph of one clip with the nodes and edges of every agent."""
    parts = [EvidenceGraph(path).records() for path in sorted(Path(graphs_root).glob(f"*/{cid}.jsonl"))]
    graph = EvidenceGraph()
    for wanted in (Node, Edge, Veracity):  # nodes of all agents first, so cross-agent edges resolve
        for part in parts:
            for record in part:
                if isinstance(record, wanted):
                    graph.add(record)
    return graph


def build(scans_dir, graphs_root, stems=None):
    scans_dir, graphs_root = Path(scans_dir), Path(graphs_root)
    out_dir = graphs_root / PRODUCER
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = ap.load_scans(scans_dir)
    labels = {int(k): v for k, v in json.loads((scans_dir / "labels.json").read_text()).items()}
    mapping, counts = [], []
    for stem in sorted(stems or clips):
        data, _ = clips[stem]
        cid = clip_id(stem)
        path = out_dir / f"{cid}.jsonl"
        path.write_text("")  # an empty graph still says the clip was processed
        graph = EvidenceGraph(path)
        nodes = event_nodes(cid, data["starts"], data["scores"].astype(np.float32), labels,
                            float(data["duration"]))
        for node in nodes:
            graph.add(node)
        mapping.append((cid, stem + ".mp4"))
        counts.append(len(nodes))
    with open(graphs_root / "clips.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "file"])
        writer.writerows(mapping)
    return {"clips": len(mapping), "nodes": int(sum(counts)), "median_nodes": float(np.median(counts))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scans", required=True, help="output folder of audio_precheck.py scan")
    parser.add_argument("--out", required=True, help="graphs root; this agent writes <out>/audio_events/")
    args = parser.parse_args(argv)
    print(build(args.scans, args.out))


if __name__ == "__main__":
    main()
