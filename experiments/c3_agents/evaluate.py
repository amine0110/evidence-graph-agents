"""Score C3 system runs against the XD-Violence labels and the human segments.

Runs on the Mac, the only place that holds clips.csv (clip id -> file name and
label); the agents never see it. Each run is a JSONL file (reasoning.py) or a
folder of per-clip JSON files (s1_omni.py) with clip_id, score, decision,
segments and, where available, cited and usage.

Per run: AUC and average precision of the score; F1, precision and recall of
the decision (an unparsed decision counts as No); triage precision and recall
when an operator reviews the top 10% or 20% of clips by score; frame AP on
one-second steps (a second gets the clip score if it lies in a predicted
segment, or if the system gave no segment, else 0); grounding precision and
recall of the predicted segments against the frame labels (24 fps) on violent
clips the system called Yes; cost. 95% CIs by bootstrap over clips.

--pair A B adds the paired difference B - A in F1 and AUC with its CI (H3a,
H3c). --faith BASE CITED RANDOM adds the decision flip rates when the cited
nodes or as many random nodes are hidden (H3b).
"""

import argparse
import csv
import json
import re
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

FPS = 24
SEED, RESAMPLES = 2026, 2000


def load_run(path):
    path = Path(path)
    if path.is_dir():
        rows = [json.loads(p.read_text()) for p in sorted(path.glob("x*.json"))]
    else:
        rows = [json.loads(l) for l in open(path) if l.strip()]
    return {r["clip_id"]: r for r in rows}


def load_truth(graphs_root, annotations, precheck):
    clips = {r["clip_id"]: r["file"] for r in csv.DictReader(open(Path(graphs_root) / "clips.csv"))}
    frames, raw = {}, {}
    for line in open(annotations):
        parts = line.split()
        if parts:
            nums = list(map(int, parts[1:]))
            frames[parts[0]] = [(a / FPS, b / FPS) for a, b in zip(nums[::2], nums[1::2])]
            raw[parts[0]] = list(zip(nums[::2], nums[1::2]))
    sound, duration = {}, {}
    if precheck:
        for r in csv.DictReader(open(precheck)):
            sound[r["file"]] = r["category"] in ("speech", "other sound")
            duration[r["file"]] = float(r["duration"])
    truth = {}
    for cid, file in clips.items():
        stem = file[:-4]
        truth[cid] = {"file": file, "violent": not re.search(r"label_A\.mp4$", file),
                      "intervals": frames.get(stem, []), "with_sound": sound.get(file, True),
                      "duration": duration.get(file, 0.0), "frame_intervals": raw.get(stem, [])}
    return truth


def seconds(intervals):
    return {s for a, b in intervals for s in range(int(a), int(np.ceil(b)))}


def f1(y, d):
    tp = int(np.sum(y & d)); fp = int(np.sum(~y & d)); fn = int(np.sum(y & ~d))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def triage(y, s, share):
    k = max(1, int(round(share * len(s))))
    order = np.lexsort((np.random.default_rng(SEED).random(len(s)), -s))  # ties broken at random, fixed seed
    top = np.zeros(len(s), bool)
    top[order[:k]] = True
    return float(np.sum(y & top) / k), float(np.sum(y & top) / max(np.sum(y), 1))


def boot(fn, n, rng):
    vals = []
    for _ in range(RESAMPLES):
        idx = rng.integers(0, n, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                v = fn(idx)
            except ValueError:
                continue
        if not np.isnan(v):
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def clip_segments(run, truth):
    """Cut every predicted segment to [0, clip length]; drop what falls outside."""
    for c, r in run.items():
        if c in truth and truth[c]["duration"]:
            d = truth[c]["duration"]
            r["segments"] = [(max(0.0, a), min(b, d)) for a, b in r.get("segments", []) if a < d]
    return run


def frame_ap(run, truth, cids, frame_counts):
    """Standard XD-Violence frame-level AP: every frame of every clip, at the video frame rate.

    A frame gets the clip score if it lies in a predicted segment, or if the system gave
    no segment; otherwise 0. Ground truth: the frame intervals of annotations.txt.
    """
    ys, ss = [], []
    for c in cids:
        info = frame_counts[truth[c]["file"]]
        n, fps = info["frames"], info["fps"]
        y = np.zeros(n, bool)
        for a, b in truth[c]["frame_intervals"]:
            y[a:b] = True
        r, score = run[c], float(run[c]["score"])
        segs = r.get("segments", [])
        if segs:
            sc = np.zeros(n)
            for a, b in segs:
                sc[int(a * fps):int(np.ceil(b * fps))] = score
        else:
            sc = np.full(n, score)
        ys.append(y); ss.append(sc)
    return round(float(average_precision_score(np.concatenate(ys), np.concatenate(ss))), 4)


def metrics(run, truth, subset=None, frame_counts=None):
    cids = sorted(c for c in run if c in truth and (subset is None or subset(truth[c])))
    y = np.array([truth[c]["violent"] for c in cids])
    s = np.array([float(run[c]["score"]) for c in cids])
    d = np.array([run[c].get("decision") == "yes" for c in cids])
    rng = np.random.default_rng(SEED)
    out = {"clips": len(cids), "auc": round(roc_auc_score(y, s), 4),
           "auc_ci": boot(lambda i: roc_auc_score(y[i], s[i]), len(cids), rng),
           "ap": round(average_precision_score(y, s), 4)}
    out["f1"], out["precision"], out["recall"] = (round(v, 4) for v in f1(y, d))
    out["f1_ci"] = boot(lambda i: f1(y[i], d[i])[0], len(cids), rng)
    out["unparsed"] = int(sum(run[c].get("decision") is None for c in cids))
    for share in (0.1, 0.2):
        p, r = triage(y, s, share)
        out[f"triage{int(share * 100)}_precision"], out[f"triage{int(share * 100)}_recall"] = round(p, 4), round(r, 4)
    # frame AP and grounding, one-second steps
    fs, fy, g_prec, g_rec = [], [], [], []
    for c, score in zip(cids, s):
        r, t = run[c], truth[c]
        length = int(np.ceil(max([b for _, b in t["intervals"]] + [b for _, b in r.get("segments", [])] + [1])))
        length = max(length, int(np.ceil(t["duration"])))
        gt = seconds(t["intervals"]) if t["violent"] else set()
        pred = seconds(r.get("segments", []))
        for sec in range(length):
            fy.append(sec in gt)
            fs.append(score if (sec in pred or not pred) else 0.0)
        if t["violent"] and r.get("decision") == "yes" and pred and gt:
            g_prec.append(len(pred & gt) / len(pred))
            g_rec.append(len(pred & gt) / len(gt))
    out["frame_ap_1s"] = round(average_precision_score(fy, fs), 4)
    if frame_counts:
        out["frame_ap"] = frame_ap(run, truth, cids, frame_counts)
        # the same with every frame of a clip given the clip score: what the segments add
        flat = {c: {**run[c], "segments": []} for c in cids}
        out["frame_ap_no_segments"] = frame_ap(flat, truth, cids, frame_counts)
    out["grounding_precision"] = round(float(np.mean(g_prec)), 4) if g_prec else None
    out["grounding_recall"] = round(float(np.mean(g_rec)), 4) if g_rec else None
    out["grounded_clips"] = len(g_prec)
    usage = [run[c].get("usage") for c in cids if run[c].get("usage")]
    if usage:
        out["tokens_per_clip"] = round(float(np.mean([u["prompt_tokens"] + u["completion_tokens"] for u in usage])), 1)
        out["calls_per_clip"] = round(float(np.mean([u["calls"] for u in usage])), 2)
    elif any("tokens" in run[c] for c in cids):
        out["tokens_per_clip"] = round(float(np.mean([run[c].get("tokens", 0) for c in cids])), 1)
    out["seconds_per_clip"] = round(float(np.mean([run[c].get("seconds", 0) for c in cids])), 2)
    out["cited_per_clip"] = round(float(np.mean([len(run[c].get("cited", [])) for c in cids])), 2)
    return out


def load_human(path, truth):
    """Annotator decisions and segments on their clips (first showing; Unsure clips dropped)."""
    by_file = {t["file"]: cid for cid, t in truth.items()}
    out = {}
    for r in map(json.loads, open(path)):
        cid = by_file.get(r.get("xd_violence_file"))
        if cid and not r.get("repeat") and r["decision"] in ("yes", "no"):
            out[cid] = {"violent": r["decision"] == "yes",
                        "segments": [(s["start"], s["end"]) for s in r["segments"]]}
    return out


def human_metrics(run, human):
    cids = sorted(c for c in run if c in human)
    y = np.array([human[c]["violent"] for c in cids])
    s = np.array([float(run[c]["score"]) for c in cids])
    d = np.array([run[c].get("decision") == "yes" for c in cids])
    out = {"clips": len(cids), "auc": round(roc_auc_score(y, s), 4) if 0 < y.sum() < len(y) else None,
           "f1": round(f1(y, d)[0], 4), "accuracy": round(float(np.mean(y == d)), 4)}
    prec, rec = [], []
    for c in cids:
        gt, pred = seconds(human[c]["segments"]), seconds(run[c].get("segments", []))
        if human[c]["violent"] and run[c].get("decision") == "yes" and gt and pred:
            prec.append(len(pred & gt) / len(pred)); rec.append(len(pred & gt) / len(gt))
    out["grounding_precision"] = round(float(np.mean(prec)), 4) if prec else None
    out["grounding_recall"] = round(float(np.mean(rec)), 4) if rec else None
    out["grounded_clips"] = len(prec)
    return out


def paired(a, b, truth, subset=None):
    cids = sorted(c for c in a if c in b and c in truth and (subset is None or subset(truth[c])))
    y = np.array([truth[c]["violent"] for c in cids])
    da = np.array([a[c].get("decision") == "yes" for c in cids]); db = np.array([b[c].get("decision") == "yes" for c in cids])
    sa = np.array([a[c]["score"] for c in cids]); sb = np.array([b[c]["score"] for c in cids])
    rng = np.random.default_rng(SEED)
    return {"clips": len(cids),
            "f1_diff": round(f1(y, db)[0] - f1(y, da)[0], 4),
            "f1_diff_ci": boot(lambda i: f1(y[i], db[i])[0] - f1(y[i], da[i])[0], len(cids), rng),
            "auc_diff": round(roc_auc_score(y, sb) - roc_auc_score(y, sa), 4),
            "auc_diff_ci": boot(lambda i: roc_auc_score(y[i], sb[i]) - roc_auc_score(y[i], sa[i]), len(cids), rng)}


def faithfulness(base, cited, rand):
    out = {}
    for name, run in (("cited", cited), ("random", rand)):
        cids = [c for c in run if c in base and base[c].get("cited")]
        flips = [run[c].get("decision") != base[c].get("decision") for c in cids]
        shift = [abs(run[c]["score"] - base[c]["score"]) for c in cids]
        out[f"flip_rate_{name}"] = round(float(np.mean(flips)), 4) if flips else None
        out[f"score_shift_{name}"] = round(float(np.mean(shift)), 4) if shift else None
        out[f"clips_{name}"] = len(cids)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", help="name=path pairs")
    parser.add_argument("--graphs", required=True, help="graphs root with clips.csv")
    parser.add_argument("--annotations", required=True, help="XD-Violence annotations.txt (frame labels)")
    parser.add_argument("--precheck", help="audio pre-check clips.csv, for the with-sound subset")
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"))
    parser.add_argument("--faith", nargs=3, action="append", default=[], metavar=("BASE", "CITED", "RANDOM"))
    parser.add_argument("--frames", help="JSON {file: {frames, fps}} for the standard frame-level AP")
    parser.add_argument("--human", nargs=2, action="append", default=[], metavar=("NAME", "JSONL"),
                        help="annotator decisions (Hugging Face annotations file) as a second reference")
    parser.add_argument("--out", help="write all results as JSON here")
    args = parser.parse_args(argv)
    truth = load_truth(args.graphs, args.annotations, args.precheck)
    runs = {name: clip_segments(load_run(path), truth) for name, path in (r.split("=", 1) for r in args.runs)}
    frame_counts = json.loads(Path(args.frames).read_text()) if args.frames else None
    with_sound = lambda t: t["with_sound"]
    result = {"runs": {}, "pairs": {}, "faithfulness": {}}
    humans = {name: load_human(path, truth) for name, path in args.human}
    for name, run in runs.items():
        result["runs"][name] = {"all": metrics(run, truth, frame_counts=frame_counts),
                                "with_sound": metrics(run, truth, with_sound)}
        for h, ref in humans.items():
            result["runs"][name][f"vs_{h}"] = human_metrics(run, ref)
            # the same clips scored against the labels, to compare rankings under both references
            result["runs"][name][f"labels_on_{h}_clips"] = human_metrics(
                run, {c: {"violent": truth[c]["violent"], "segments": truth[c]["intervals"]} for c in ref})
    for a, b in args.pair:
        result["pairs"][f"{b} - {a}"] = {"all": paired(runs[a], runs[b], truth),
                                        "with_sound": paired(runs[a], runs[b], truth, with_sound)}
    for base, cited, rand in args.faith:
        result["faithfulness"][base] = faithfulness(runs[base], runs[cited], runs[rand])
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
