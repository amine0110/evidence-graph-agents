"""Video agent, step 2: turn detections and captions into graph nodes.

detection nodes: one per object and run of seconds in which the detector saw
it (gaps of up to MAX_GAP seconds are bridged), with the most instances seen
in one frame and the peak score; the peak score is the node's confidence.
caption nodes: one per 4-second caption. The VLM gives no calibrated score,
so caption nodes carry confidence 1.0 and the reasoning agent must judge the
text itself.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evgraph import EvidenceGraph, Modality, Node, Source

PRODUCER = "video"
MAX_GAP = 1


def object_runs(detections):
    """{(detector, object): [(first second, last second, {second: [conf, ...]})]}"""
    seconds = defaultdict(lambda: defaultdict(list))
    for d in detections:
        seconds[(d.get("model", ""), d["object"])][d["t"]].append(d["conf"])
    runs = {}
    for obj, by_second in seconds.items():
        ts, groups, current = sorted(by_second), [], []
        for t in ts:
            if current and t - current[-1] > MAX_GAP + 1:
                groups.append(current)
                current = []
            current.append(t)
        if current:
            groups.append(current)
        runs[obj] = [(g[0], g[-1], {t: by_second[t] for t in g}) for g in groups]
    return runs


def video_nodes(record):
    cid, nodes = record["clip_id"], []
    for (model, obj), runs in sorted(object_runs(record["detections"]).items()):
        for first, last, confs in runs:
            peak_t = max(confs, key=lambda t: max(confs[t]))
            peak = max(confs[peak_t])
            nodes.append(Node(
                id=f"{cid}:video:{obj.replace(' ', '_')}:{len(nodes)}",
                modality=Modality.VIDEO, kind="detection",
                content={"object": obj, "max_count": max(len(c) for c in confs.values()),
                         "seconds_seen": len(confs), "peak_score": round(peak, 4), "peak_time": peak_t},
                t_obs=(float(first), float(last + 1)), t_rec=float(last + 1),
                source=Source(producer=PRODUCER, version=record["detectors"].get(model, model), input_ref=cid,
                              offset=f"{first}-{last + 1}s"),
                confidence=round(peak, 4)))
    for k, cap in enumerate(record["captions"]):
        if not cap["text"]:
            continue
        nodes.append(Node(
            id=f"{cid}:video:caption:{k}", modality=Modality.VIDEO, kind="caption",
            content={"text": cap["text"]},
            t_obs=(float(cap["start"]), float(cap["end"])), t_rec=float(cap["end"]),
            source=Source(producer=PRODUCER, version=record["vlm"], input_ref=cid,
                          offset=f"{cap['start']}-{cap['end']}s"),
            confidence=1.0))
    return sorted(nodes, key=lambda n: (n.t_obs[0], n.id))


def build(video_dir, graphs_root):
    out_dir = Path(graphs_root) / PRODUCER
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = detections = captions = 0
    for path in sorted(Path(video_dir).glob("x*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        target = out_dir / f"{record['clip_id']}.jsonl"
        target.write_text("")
        graph = EvidenceGraph(target)
        for node in video_nodes(record):
            graph.add(node)
            detections += node.kind == "detection"
            captions += node.kind == "caption"
        clips += 1
    return {"clips": clips, "detection_nodes": detections, "caption_nodes": captions}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True, help="output folder of video_pass.py")
    parser.add_argument("--out", required=True, help="graphs root; this agent writes <out>/video/")
    args = parser.parse_args(argv)
    print(build(args.video, args.out))


if __name__ == "__main__":
    main()
