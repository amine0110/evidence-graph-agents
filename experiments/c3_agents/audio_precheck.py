"""Audio pre-check for C3 Domain A (XD-Violence test set).

Answers, before any agent is built: which clips hold real sound (speech or
other sound) rather than near silence or music only, whether that is spread
evenly across classes (a silence shortcut), and whether an off-the-shelf
AudioSet tagger hears what the human annotators said they heard.

scan    decodes each clip's audio (16 kHz mono), measures loudness per second
        and runs an AudioSet tagger (AST) on 10 s windows every 5 s. One
        .npz per clip, so an interrupted scan resumes where it stopped.
report  turns the scans into clip categories and the summary tables.

Thresholds are fixed here, before looking at the results:
    near-silent   fewer than 10% of seconds louder than -50 dBFS
    speech        "speech" group score >= 0.5 in at least 10% of windows
    music only    "Music" >= 0.5 in at least 80% of windows and not speech
    with sound    everything else (the H3c subset: speech or other sound)
    event heard   group score >= 0.3 in a window overlapping the segment
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000
WIN, HOP = 10.0, 5.0
SILENT_DBFS, MIN_ACTIVE = -50.0, 0.10
SPEECH_MIN, MUSIC_MIN, EVENT_MIN = 0.5, 0.5, 0.3
SPEECH_SHARE, MUSIC_SHARE = 0.10, 0.80

GROUPS = {
    "speech": ["Speech", "Male speech, man speaking", "Female speech, woman speaking",
               "Child speech, kid speaking", "Conversation", "Narration, monologue"],
    "music": ["Music"],
    "gunshot": ["Gunshot, gunfire", "Machine gun", "Artillery fire"],
    "explosion": ["Explosion", "Boom"],
    "scream": ["Screaming", "Shout", "Yell", "Battle cry", "Crying, sobbing", "Children shouting"],
    "crash": ["Smash, crash", "Shatter", "Breaking", "Skidding", "Tire squeal", "Bang"],
    "crowd": ["Crowd", "Hubbub, speech noise, speech babble"],
    "siren": ["Siren", "Police car (siren)", "Ambulance (siren)",
              "Fire engine, fire truck (siren)", "Civil defense siren"],
    "impact": ["Slap, smack", "Whack, thwack"],
}
EVENT_GROUPS = ["gunshot", "explosion", "scream", "crash", "crowd", "siren", "impact"]
CLASS_NAMES = {"A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot",
               "B5": "abuse", "B6": "car accident", "G": "explosion"}


def decode(path):
    import av
    with av.open(str(path)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def loudness(wave):
    """RMS level in dBFS for each whole second."""
    n = len(wave) // SR
    if n == 0:
        return np.array([-120.0])
    rms = np.sqrt(np.mean(wave[:n * SR].reshape(n, SR) ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-6))


def windows(wave):
    starts = np.arange(0, max(len(wave) / SR - WIN, 0) + 1e-9, HOP)
    return starts, [wave[int(s * SR):int((s + WIN) * SR)] for s in starts]


def scan(videos_root, out_dir, batch=16, device=None):
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    extractor = ASTFeatureExtractor.from_pretrained(MODEL)
    model = ASTForAudioClassification.from_pretrained(MODEL).eval().to(device)
    out_dir = Path(out_dir)
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.json").write_text(json.dumps(model.config.id2label))
    videos = sorted(Path(videos_root).glob("*.mp4"))
    t0 = time.time()
    for i, video in enumerate(videos, 1):
        target = out_dir / "clips" / (video.stem + ".npz")
        if target.exists():
            continue
        wave = decode(video)
        starts, parts = windows(wave)
        scores = []
        for k in range(0, len(parts), batch):
            x = extractor(parts[k:k + batch], sampling_rate=SR, return_tensors="pt")["input_values"]
            with torch.no_grad():
                scores.append(torch.sigmoid(model(x.to(device)).logits).cpu().numpy())
        scores = np.concatenate(scores) if scores else np.zeros((0, model.config.num_labels))
        np.savez_compressed(target, dbfs=loudness(wave).astype(np.float32),
                            starts=starts.astype(np.float32), scores=scores.astype(np.float16),
                            duration=np.float32(len(wave) / SR))
        if i % 25 == 0:
            print(f"{i}/{len(videos)} {time.time() - t0:.0f}s", flush=True)


def label_of(stem):
    return stem.split("_label_")[1].split("-")[0]


def group_scores(scores, labels):
    index = {name: i for i, name in labels.items()}
    return {g: scores[:, [index[n] for n in names if n in index]].max(axis=1)
            if len(scores) else np.zeros(0) for g, names in GROUPS.items()}


def categorise(clip, groups):
    active = float(np.mean(clip["dbfs"] > SILENT_DBFS))
    speech = float(np.mean(groups["speech"] >= SPEECH_MIN)) if len(groups["speech"]) else 0.0
    music = float(np.mean(groups["music"] >= MUSIC_MIN)) if len(groups["music"]) else 0.0
    if active < MIN_ACTIVE:
        category = "near-silent"
    elif speech >= SPEECH_SHARE:
        category = "speech"
    elif music >= MUSIC_SHARE:
        category = "music only"
    else:
        category = "other sound"
    return {"active_share": round(active, 3), "speech_share": round(speech, 3),
            "music_share": round(music, 3), "median_dbfs": round(float(np.median(clip["dbfs"])), 1),
            "category": category, "with_sound": category in ("speech", "other sound")}


def load_scans(out_dir):
    out_dir = Path(out_dir)
    labels = {int(k): v for k, v in json.loads((out_dir / "labels.json").read_text()).items()}
    clips = {}
    for path in sorted((out_dir / "clips").glob("*.npz")):
        data = dict(np.load(path))
        clips[path.stem] = (data, group_scores(data["scores"].astype(np.float32), labels))
    return clips


def window_hits(data, groups, start, end, group):
    starts = data["starts"]
    overlap = (starts < end) & (starts + WIN > start)
    if not overlap.any() and len(starts):
        overlap = np.abs(starts + WIN / 2 - (start + end) / 2) == np.min(np.abs(starts + WIN / 2 - (start + end) / 2))
    return float(groups[group][overlap].max()) if overlap.any() else 0.0


def human_items(annotator_dir):
    """First showing of each clip: (filename stem, latest record)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "video-review-annotator"))
    from annotator.analyze import load_latest
    annotator_dir = Path(annotator_dir)
    manifest = json.loads((annotator_dir / "manifest.json").read_text())["items"]
    latest = load_latest(annotator_dir / "annotations.jsonl")
    return {Path(item["path"]).stem: latest[i] for i, item in enumerate(manifest)
            if not item.get("repeat") and i in latest}


def report(out_dir, annotator_dirs):
    out_dir = Path(out_dir)
    clips = load_scans(out_dir)
    rows = []
    for stem, (data, groups) in clips.items():
        row = {"file": stem + ".mp4", "label": label_of(stem), "class": CLASS_NAMES.get(label_of(stem), "?"),
               "duration": round(float(data["duration"]), 1)}
        row.update(categorise(data, groups))
        for g in EVENT_GROUPS:
            row[g] = round(float(groups[g].max()) if len(groups[g]) else 0.0, 3)
        rows.append(row)
    with open(out_dir / "clips.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"clips": len(rows), "thresholds": {
        "silent_dbfs": SILENT_DBFS, "min_active": MIN_ACTIVE, "speech_min": SPEECH_MIN,
        "speech_share": SPEECH_SHARE, "music_min": MUSIC_MIN, "music_share": MUSIC_SHARE,
        "event_min": EVENT_MIN, "window_s": WIN, "hop_s": HOP, "model": MODEL}}
    categories = ["speech", "other sound", "music only", "near-silent"]
    summary["by_category"] = {c: sum(r["category"] == c for r in rows) for c in categories}
    by_class = {}
    for r in rows:
        by_class.setdefault(r["class"], {c: 0 for c in categories})[r["category"]] += 1
    summary["by_class"] = by_class
    violent = [r for r in rows if r["label"] != "A"]
    normal = [r for r in rows if r["label"] == "A"]
    summary["with_sound_share"] = {
        "violent": round(np.mean([r["with_sound"] for r in violent]), 3),
        "normal": round(np.mean([r["with_sound"] for r in normal]), 3)}
    from scipy.stats import fisher_exact
    table = [[sum(r["with_sound"] for r in violent), sum(not r["with_sound"] for r in violent)],
             [sum(r["with_sound"] for r in normal), sum(not r["with_sound"] for r in normal)]]
    summary["silence_shortcut"] = {"table_violent_normal_by_sound": table,
                                   "fisher_p": float(fisher_exact(table)[1])}

    by_file = {r["file"][:-4]: r for r in rows}
    humans = {}
    for annotator_dir in annotator_dirs:
        items = human_items(annotator_dir)
        yes = [(stem, rec) for stem, rec in items.items() if rec["decision"] == "yes" and stem in by_file]
        def reliance(subset):
            used = [set().union(*[set(s.get("modalities", [])) for s in rec["segments"]]) for _, rec in subset]
            return {"n_yes": len(subset),
                    "sound_or_speech": round(np.mean([bool(u & {"sound", "speech"}) for u in used]), 3) if used else None,
                    "sound": round(np.mean(["sound" in u for u in used]), 3) if used else None,
                    "speech": round(np.mean(["speech" in u for u in used]), 3) if used else None}
        with_sound = [(s, r) for s, r in yes if by_file[s]["with_sound"]]
        heard = {"sound_ticked": [], "sound_not_ticked": [], "speech_ticked": [], "speech_not_ticked": []}
        for stem, rec in yes:
            data, groups = clips[stem]
            for seg in rec["segments"]:
                events = max(window_hits(data, groups, seg["start"], seg["end"], g) for g in EVENT_GROUPS)
                speech = window_hits(data, groups, seg["start"], seg["end"], "speech")
                mods = set(seg.get("modalities", []))
                heard["sound_ticked" if "sound" in mods else "sound_not_ticked"].append(events >= EVENT_MIN)
                heard["speech_ticked" if "speech" in mods else "speech_not_ticked"].append(speech >= SPEECH_MIN)
        humans[Path(annotator_dir).name] = {
            "all_yes": reliance(yes), "with_sound_yes": reliance(with_sound),
            "yes_clips_not_with_sound": sorted({by_file[s]["category"] for s, _ in yes} - {"speech", "other sound"}),
            "segments_detector_agrees": {k: {"n": len(v), "share": round(float(np.mean(v)), 3) if v else None}
                                         for k, v in heard.items()}}
    summary["humans"] = humans
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--videos-root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device")
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    r.add_argument("--annotator", action="append", default=[], help="folder with manifest.json and annotations.jsonl")
    args = parser.parse_args(argv)
    if args.command == "scan":
        scan(args.videos_root, args.out, device=args.device)
    else:
        report(args.out, args.annotator)


if __name__ == "__main__":
    main()
