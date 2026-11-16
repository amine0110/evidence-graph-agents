"""Freeze one worked example: the evidence of a clip and every system's answer on it."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import reasoning  # noqa: E402
from audio_events import clip_graph  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument("--graphs", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    runs = Path(a.runs)
    view = reasoning.View(clip_graph(a.graphs, a.clip))
    ex = {"clip_id": a.clip, "duration": view.duration, "evidence": [view.line(n) for n in view.nodes], "systems": {}}
    for name in ("s2", "s3", "s4"):
        r = next(json.loads(l) for l in open(runs / f"{name}.jsonl") if json.loads(l)["clip_id"] == a.clip)
        ex["systems"][name] = {"score": r["score"], "answer": r["answer"], "trace": r.get("trace", [])}
    for name in ("s1_omni", "s1_ts"):
        r = json.loads((runs / name / f"{a.clip}.json").read_text())
        ex["systems"][name] = {"score": r["score"], "answer": r["answer"]}
    Path(a.out).write_text(json.dumps(ex, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
