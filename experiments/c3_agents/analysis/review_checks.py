"""Checks asked for in review, all from the frozen runs (no new model calls).

1. Evidence size: split the test clips into thirds by the number of graph
   nodes (what S4 reads) and give each system's F1 and recall per third, with
   paired F1 differences of S4 against S1, S2 and S3 v1 (95% bootstrap CIs).
2. Grounding on a common set: precision and recall of the predicted segments on
   the violent clips that every system called Yes, so all systems are scored
   on the same clips.
3. Faithfulness: cited nodes per clip and a bootstrap CI for the difference
   in cited-node flip rates between S4 and S3 v1.
4. Larger reasoning model: S3 and S4 with Qwen3.6 on the clips where both
   answers could be parsed.

Runs on the Mac (needs clips.csv for the labels). Writes frozen/review_checks.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from evaluate import SEED, boot, clip_segments, f1, load_run, load_truth, seconds  # noqa: E402

MAIN = (("s1", "s1_omni"), ("s1_ts", "s1_ts"), ("s2", "s2.jsonl"), ("s3", "s3.jsonl"), ("s3v2", "s3v2.jsonl"),
        ("s4", "s4.jsonl"))


def r4(x):
    return round(float(x), 4)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(Path.home() / "data/XD-Violence"))
    parser.add_argument("--out", default=str(HERE.parent / "frozen/review_checks.json"))
    args = parser.parse_args(argv)
    data = Path(args.data)
    truth = load_truth(data / "graphs", data / "annotations.txt", data / "precheck/clips.csv")
    runs = {k: clip_segments(load_run(data / "runs" / p), truth) for k, p in MAIN}
    cids = sorted(c for c in truth if all(c in r for r in runs.values()))
    y = np.array([truth[c]["violent"] for c in cids])
    dec = {k: np.array([r[c].get("decision") == "yes" for c in cids]) for k, r in runs.items()}
    out = {}

    # 1. evidence size
    nodes = np.array([runs["s4"][c]["nodes_visible"] for c in cids])
    tokens = np.array([runs["s4"][c]["usage"]["prompt_tokens"] for c in cids])
    cuts = np.percentile(nodes, [100 / 3, 200 / 3])
    third = np.digitize(nodes, cuts, right=True)
    size = {"nodes_median": int(np.median(nodes)), "nodes_max": int(nodes.max()),
            "s4_prompt_tokens_median": int(np.median(tokens)), "s4_prompt_tokens_max": int(tokens.max()),
            "cuts": [float(c) for c in cuts], "thirds": []}
    for t in range(3):
        sub = third == t
        ys = y[sub]
        row = {"clips": int(sub.sum()), "violent": int(ys.sum()), "nodes_min": int(nodes[sub].min()),
               "nodes_max": int(nodes[sub].max()), "runs": {}, "pairs": {}}
        for k in ("s1", "s2", "s3", "s4"):
            fv, p, r = f1(ys, dec[k][sub])
            row["runs"][k] = {"f1": r4(fv), "recall": r4(r), "precision": r4(p)}
        for b in ("s1", "s2", "s3"):
            da, db = dec["s4"][sub], dec[b][sub]
            rng = np.random.default_rng(SEED)
            row["pairs"][f"s4 - {b}"] = {"f1_diff": r4(f1(ys, da)[0] - f1(ys, db)[0]),
                                         "f1_diff_ci": boot(lambda i: f1(ys[i], da[i])[0] - f1(ys[i], db[i])[0],
                                                            len(ys), rng)}
        size["thirds"].append(row)
    out["size"] = size

    # 2. grounding on the violent clips every system called Yes
    common = [c for c, v in zip(cids, y) if v and all(r[c].get("decision") == "yes" for r in runs.values())]
    g = {"clips": len(common), "runs": {}}
    for k, r in runs.items():
        prec, rec = [], []
        for c in common:
            gt, pred = seconds(truth[c]["intervals"]), seconds(r[c].get("segments", []))
            if pred and gt:
                prec.append(len(pred & gt) / len(pred)); rec.append(len(pred & gt) / len(gt))
        g["runs"][k] = {"precision": r4(np.mean(prec)), "recall": r4(np.mean(rec)), "with_segments": len(prec)}
    out["grounding_common"] = g

    # 3. faithfulness
    faith = {}
    flips = {}
    for k in ("s3", "s4"):
        base = load_run(data / "runs" / f"{k}.jsonl")
        cited = load_run(data / "runs" / f"{k}_del_cited.jsonl")
        ids = [c for c in cited if c in base and base[c].get("cited")]
        flips[k] = np.array([cited[c].get("decision") != base[c].get("decision") for c in ids])
        faith[k] = {"clips": len(ids), "flip_rate": r4(flips[k].mean()),
                    "cited_per_clip": r4(np.mean([len(base[c]["cited"]) for c in ids])),
                    "cited_median": float(np.median([len(base[c]["cited"]) for c in ids]))}
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(2000):
        a = flips["s4"][rng.integers(0, len(flips["s4"]), len(flips["s4"]))]
        b = flips["s3"][rng.integers(0, len(flips["s3"]), len(flips["s3"]))]
        diffs.append(a.mean() - b.mean())
    faith["s4 - s3"] = {"diff": r4(flips["s4"].mean() - flips["s3"].mean()),
                        "ci": [r4(np.percentile(diffs, 2.5)), r4(np.percentile(diffs, 97.5))]}
    out["faithfulness"] = faith

    # 4. Qwen3.6 on clips where both S3 and S4 answers were parsed
    q3, q4 = load_run(data / "runs/s3_qwen36.jsonl"), load_run(data / "runs/s4_qwen36.jsonl")
    ok = [c for c in cids if c in q3 and c in q4 and q3[c].get("decision") is not None
          and q4[c].get("decision") is not None]
    yq = np.array([truth[c]["violent"] for c in ok])
    d3 = np.array([q3[c]["decision"] == "yes" for c in ok]); d4 = np.array([q4[c]["decision"] == "yes" for c in ok])
    rng = np.random.default_rng(SEED)
    out["qwen_parsed"] = {"clips": len(ok), "s3_f1": r4(f1(yq, d3)[0]), "s4_f1": r4(f1(yq, d4)[0]),
                          "s3_minus_s4": r4(f1(yq, d3)[0] - f1(yq, d4)[0]),
                          "s3_minus_s4_ci": boot(lambda i: f1(yq[i], d3[i])[0] - f1(yq[i], d4[i])[0], len(ok), rng)}

    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
