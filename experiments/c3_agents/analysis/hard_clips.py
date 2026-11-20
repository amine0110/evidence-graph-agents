"""Where the systems differ, on a test set near its ceiling.

Two views that do not depend on the overall F1, both from the frozen runs:

1. Discordant clips. For each pair of systems, the clips on which their
   decisions differ and how many each gets right, with an exact two-sided
   McNemar (binomial) test. Overall F1 near 0.97 hides how many clips separate
   two systems; this counts them.
2. Short incidents. The quarter of violent clips whose labelled violence covers
   the smallest share of the clip (fixed by the labels alone, before looking at
   any system), together with all normal clips. F1 and recall per system and
   paired F1 differences with 95% bootstrap intervals, as in evaluate.py.

Runs on the Mac (needs clips.csv for the labels). Writes frozen/hard_clips.json.
"""

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from evaluate import SEED, boot, f1, load_run, load_truth  # noqa: E402

SYSTEMS = (("s1", "s1_omni"), ("s2", "s2.jsonl"), ("s3", "s3.jsonl"), ("s3v2", "s3v2.jsonl"), ("s4", "s4.jsonl"))
PAIRS = (("s4", "s1"), ("s4", "s2"), ("s4", "s3"), ("s4", "s3v2"), ("s3", "s2"))


def mcnemar(a_right, b_right):
    """Exact two-sided McNemar p-value on the discordant counts."""
    n, k = a_right + b_right, min(a_right, b_right)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(Path.home() / "data/XD-Violence"))
    parser.add_argument("--out", default=str(HERE.parent / "frozen/hard_clips.json"))
    args = parser.parse_args(argv)
    data = Path(args.data)
    truth = load_truth(data / "graphs", data / "annotations.txt", data / "precheck/clips.csv")
    runs = {k: load_run(data / "runs" / p) for k, p in SYSTEMS}
    cids = sorted(c for c in truth if all(c in r for r in runs.values()))
    y = np.array([truth[c]["violent"] for c in cids])
    dec = {k: np.array([r[c].get("decision") == "yes" for c in cids]) for k, r in runs.items()}

    out = {"clips": len(cids), "errors": {}, "discordant": {}, "short": {}}
    for k, d in dec.items():
        out["errors"][k] = {"errors": int(np.sum(d != y)), "false_pos": int(np.sum(d & ~y)),
                            "false_neg": int(np.sum(~d & y))}
    for a, b in PAIRS:
        dis = dec[a] != dec[b]
        ar, br = int(np.sum(dis & (dec[a] == y))), int(np.sum(dis & (dec[b] == y)))
        out["discordant"][f"{a} - {b}"] = {"clips": int(dis.sum()), "a_right": ar, "b_right": br,
                                           "p_mcnemar": float(f"{mcnemar(ar, br):.3g}")}

    share = np.array([sum(e - s for s, e in truth[c]["intervals"]) / truth[c]["duration"] if truth[c]["duration"]
                      else 1.0 for c in cids])
    cut = float(np.percentile(share[y], 25))
    sub = ~y | (share < cut)
    ys = y[sub]
    out["short"] = {"share_cut": round(cut, 4), "violent": int(np.sum(ys)), "normal": int(np.sum(~ys)), "runs": {},
                    "pairs": {}}
    for k, d in dec.items():
        ds = d[sub]
        rng = np.random.default_rng(SEED)
        fv, p, r = f1(ys, ds)
        out["short"]["runs"][k] = {"f1": round(fv, 4), "precision": round(p, 4), "recall": round(r, 4),
                                   "f1_ci": boot(lambda i: f1(ys[i], ds[i])[0], len(ys), rng)}
    for a, b in PAIRS:
        da, db = dec[a][sub], dec[b][sub]
        rng = np.random.default_rng(SEED)
        out["short"]["pairs"][f"{a} - {b}"] = {
            "f1_diff": round(f1(ys, da)[0] - f1(ys, db)[0], 4),
            "f1_diff_ci": boot(lambda i: f1(ys[i], da[i])[0] - f1(ys[i], db[i])[0], len(ys), rng)}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
