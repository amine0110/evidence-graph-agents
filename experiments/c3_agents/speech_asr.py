"""Speech agent, step 1: transcribe each clip with Whisper (GPU job).

Writes one JSON per clip, named by its neutral clip id (see audio_events.clip_id),
with the detected language and the timed segments. Decoding is deterministic
(temperature 0, no conditioning on earlier text, which also avoids repetition
loops), and Whisper's voice-activity filter skips silence and music.
Existing outputs are kept, so a stopped job resumes; --shard splits the clips
over several jobs.

Step 2 (speech.py) turns the transcripts into graph nodes.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

MODEL = "large-v3"


def clip_id(stem):
    return "x" + hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(argv)

    import faster_whisper
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = sorted(Path(args.videos_root).glob("*.mp4"))[args.shard::args.shards]
    t0 = time.time()
    for i, video in enumerate(videos, 1):
        target = out / f"{clip_id(video.stem)}.json"
        if target.exists():
            continue
        start = time.time()
        segments, info = model.transcribe(
            str(video), beam_size=5, temperature=0.0, condition_on_previous_text=False,
            vad_filter=True)
        rows = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                 "avg_logprob": round(s.avg_logprob, 4), "no_speech_prob": round(s.no_speech_prob, 4),
                 "compression_ratio": round(s.compression_ratio, 3)} for s in segments]
        record = {"clip_id": clip_id(video.stem), "model": f"faster-whisper {faster_whisper.__version__} {args.model}",
                  "language": info.language, "language_probability": round(info.language_probability, 4),
                  "duration": round(info.duration, 2), "speech_seconds": round(info.duration_after_vad, 2),
                  "seconds": round(time.time() - start, 1), "segments": rows}
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        tmp.rename(target)
        if i % 25 == 0:
            print(f"{i}/{len(videos)} {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(videos)} clips in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
