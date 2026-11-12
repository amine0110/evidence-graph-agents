"""Speech agent, step 4: turn the threat and distress tags into graph nodes.

One speech_tag node per transcript segment whose most likely label is threat,
distress or report with probability at least TAG_MIN; the probability is the
node's confidence. The node names the transcript_segment it was derived from
(content["from"]); edges.py writes the derived_from edge, because edges between
agents live in the edge builder's own files.
"""

import argparse
import json
from pathlib import Path

from evgraph import EvidenceGraph, Modality, Node, Source

PRODUCER = "speech_tags"
TAG_MIN = 0.5


def tag_nodes(record):
    nodes = []
    for seg in record["segments"]:
        tag, p = max(seg["probs"].items(), key=lambda kv: kv[1])
        if tag == "none" or p < TAG_MIN:
            continue
        nodes.append(Node(
            id=f"{seg['node']}:tag", modality=Modality.SPEECH, kind="speech_tag",
            content={"tag": tag, "text": seg["text"], "probs": seg["probs"], "from": seg["node"]},
            t_obs=(seg["start"], seg["end"]), t_rec=seg["end"],
            source=Source(producer=PRODUCER, version=record["model"], input_ref=record["clip_id"],
                          offset=f"{seg['start']:.1f}-{seg['end']:.1f}s"),
            confidence=round(min(max(p, 0.0), 1.0), 4)))
    return nodes


def build(tags_dir, graphs_root):
    out_dir = Path(graphs_root) / PRODUCER
    out_dir.mkdir(parents=True, exist_ok=True)
    clips, counts, segments = 0, {}, 0
    for path in sorted(Path(tags_dir).glob("x*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        target = out_dir / f"{record['clip_id']}.jsonl"
        target.write_text("")
        graph = EvidenceGraph(target)
        for node in tag_nodes(record):
            graph.add(node)
            counts[node.content["tag"]] = counts.get(node.content["tag"], 0) + 1
        clips += 1
        segments += len(record["segments"])
    return {"clips": clips, "segments": segments, "tag_nodes": counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tags", required=True, help="output folder of speech_tag_pass.py")
    parser.add_argument("--out", required=True, help="graphs root; this agent writes <out>/speech_tags/")
    args = parser.parse_args(argv)
    print(build(args.tags, args.out))


if __name__ == "__main__":
    main()
