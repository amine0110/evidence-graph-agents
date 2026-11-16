"""Failure analysis of S3 v1 against S4 (flat text), on the test set.

Step 1 (automatic, all clips): replay every S3 tool call offline (the tools are
deterministic) to recover which nodes S3 was shown, and check whether the nodes
S4 cited on the same clip ever reached S3.
Step 2 (manual): write a dossier for a seeded sample of clips that S3 got
wrong and S4 got right, for labelling with the categories in CATEGORIES.

Runs on the Mac (needs clips.csv for the labels).
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import reasoning  # noqa: E402
from audio_events import clip_graph  # noqa: E402

CATEGORIES = {
    "early_stop": "stopped before the decisive evidence was shown",
    "misjudged": "the decisive evidence was shown, but the decision went the other way",
    "detector_lure": "a false object detection drove a wrong Yes",
    "vocabulary": "searched for a concept the lexicon does not cover, or the evidence used words outside it",
    "label_debatable": "the clip's label is itself debatable",
}
SEED = 2026


def shown_nodes(view, trace):
    """Replay S3's tool calls; return the ids of every node its tools listed."""
    view.shown = set()
    view.overview()
    for action in trace:
        if isinstance(action, dict) and action.get("tool") not in (None, "done"):
            reasoning.call_tool(view, action)
    return set(view.shown)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample", type=int, default=50)
    args = parser.parse_args(argv)
    graphs, runs, out = Path(args.graphs), Path(args.runs), Path(args.out)
    files = {r["clip_id"]: r["file"] for r in csv.DictReader(open(graphs / "clips.csv"))}
    s3 = {json.loads(l)["clip_id"]: json.loads(l) for l in open(runs / "s3.jsonl")}
    s4 = {json.loads(l)["clip_id"]: json.loads(l) for l in open(runs / "s4.jsonl")}
    violent = {c: not re.search(r"label_A\.mp4$", f) for c, f in files.items()}

    rows = []
    for cid, r in s3.items():
        view = reasoning.View(clip_graph(graphs, cid))
        shown = shown_nodes(view, r["trace"])
        s4_cited = set(s4[cid].get("cited", []))
        right3 = (r["decision"] == "yes") == violent[cid]
        right4 = (s4[cid]["decision"] == "yes") == violent[cid]
        steps = [a.get("tool") for a in r["trace"] if isinstance(a, dict)]
        rows.append({"clip_id": cid, "violent": violent[cid], "s3_right": right3, "s4_right": right4,
                     "nodes": len(view.nodes), "shown": len(shown), "shown_share": len(shown) / max(len(view.nodes), 1),
                     "captions": sum(n.kind == "caption" for n in view.nodes),
                     "captions_shown": sum(view.graph.nodes[i].kind == "caption" for i in shown),
                     "s4_cited": len(s4_cited), "s4_cited_shown": len(s4_cited & shown),
                     "tool_calls": sum(t != "done" for t in steps), "stopped_by_itself": "done" in steps})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "s3_coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def summary(sel):
        n = len(sel)
        with_cites = [r for r in sel if r["s4_cited"]]
        return {"clips": n,
                "median_shown_share": round(sorted(r["shown_share"] for r in sel)[n // 2], 3) if n else None,
                "s4_evidence_never_shown": sum(r["s4_cited_shown"] == 0 for r in with_cites),
                "s4_evidence_partly_shown": sum(0 < r["s4_cited_shown"] < r["s4_cited"] for r in with_cites),
                "s4_evidence_all_shown": sum(r["s4_cited_shown"] == r["s4_cited"] for r in with_cites),
                "with_s4_citations": len(with_cites),
                "mean_tool_calls": round(sum(r["tool_calls"] for r in sel) / max(n, 1), 2)}
    groups = {
        "all": rows,
        "s3_wrong_s4_right": [r for r in rows if not r["s3_right"] and r["s4_right"]],
        "s3_wrong_s4_right_violent": [r for r in rows if not r["s3_right"] and r["s4_right"] and r["violent"]],
        "s3_wrong_s4_right_normal": [r for r in rows if not r["s3_right"] and r["s4_right"] and not r["violent"]],
        "both_right": [r for r in rows if r["s3_right"] and r["s4_right"]],
    }
    result = {k: summary(v) for k, v in groups.items()}
    (out / "s3_coverage_summary.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))

    # Step 2: dossiers for a seeded sample of S3-wrong, S4-right clips.
    pool = sorted(r["clip_id"] for r in groups["s3_wrong_s4_right"])
    pick = random.Random(SEED).sample(pool, min(args.sample, len(pool)))
    lines = []
    for k, cid in enumerate(pick, 1):
        r3, r4 = s3[cid], s4[cid]
        view = reasoning.View(clip_graph(graphs, cid))
        shown = shown_nodes(view, r3["trace"])
        lines.append(f"### {k}. {cid} label={'violent' if violent[cid] else 'normal'} "
                     f"S3={r3['decision']} S4={r4['decision']} nodes={len(view.nodes)} shown={len(shown)}")
        lines.append("S3 trace: " + json.dumps(r3["trace"]))
        lines.append("S3 answer: " + r3["answer"].replace("\n", " | "))
        lines.append("S3 cited: " + " / ".join(view.line(view.graph.nodes[i])[:140] for i in r3["cited"][:4]))
        lines.append("S4 answer: " + r4["answer"].replace("\n", " | "))
        for i in r4.get("cited", [])[:6]:
            if i in view.graph.nodes:
                tag = "SHOWN to S3" if i in shown else "never shown to S3"
                lines.append(f"  S4 cite ({tag}): {view.line(view.graph.nodes[i])[:200]}")
        lines.append("")
    (out / "s3_dossiers.md").write_text("\n".join(lines) + "\n")
    (out / "s3_sample.json").write_text(json.dumps(pick) + "\n")


if __name__ == "__main__":
    main()
