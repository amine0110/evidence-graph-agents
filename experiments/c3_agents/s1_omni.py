"""S1 baseline: one omni model watches and listens to the whole clip (GPU job).

The model gets the frames at 1 per second, the soundtrack at 16 kHz and the
shared task question (task.py), and answers Yes or No plus time segments. The
score is P(Yes) / (P(Yes) + P(No)) on the first answer token. No graph, no
agents: this is the "one large model" alternative to the evidence graph.

Writes one JSON per clip, named by its neutral clip id; resumable; --shard
splits the clips over several jobs.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from task import QUESTION, TASK_VERSION, parse_answer
from video_pass import clip_id, frames_per_second

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SAMPLE_RATE = 16000
MAX_FRAMES = 240  # longer clips are sampled evenly down to this many frames


def soundtrack(path):
    """Mono float32 audio at SAMPLE_RATE, or None for a clip without sound."""
    import av
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        for frame in container.decode(container.streams.audio[0]):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else None


def chunked(frames, audio, size, VideoMetadata):
    """Content items, videos, audio pieces and metadata for a clip cut into labelled chunks.

    Frames are one per second; when the clip has more than MAX_FRAMES frames, every
    step-th frame of each chunk is kept. A chunk with one frame repeats it, since the
    vision encoder groups frames in pairs.
    """
    step = max(1, int(np.ceil(len(frames) / MAX_FRAMES)))
    content, videos, pieces, metas = [], [], [], []
    for a in range(0, len(frames), size):
        b = min(a + size, len(frames))
        chunk = frames[a:b:step] or frames[a:a + 1]
        if len(chunk) == 1:
            chunk = chunk * 2
        content.append({"type": "text", "text": f"[{a}-{b} s]"})
        content.append({"type": "video"})
        videos.append(chunk)
        metas.append(VideoMetadata(total_num_frames=len(chunk), fps=len(chunk) / max(b - a, 1),
                                   duration=float(b - a), frames_indices=list(range(len(chunk)))))
        if audio is not None:
            piece = audio[a * SAMPLE_RATE:b * SAMPLE_RATE]
            if len(piece) >= SAMPLE_RATE // 2:
                content.append({"type": "audio"})
                pieces.append(piece)
    return content, videos, pieces, metas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timestamps", type=int, default=0, metavar="SECONDS",
                        help="feed the clip in chunks of this many seconds, each after a text label "
                             "'[a-b s]' with its video and sound, so the model knows absolute time (S1-ts)")
    args = parser.parse_args(argv)

    import torch
    import transformers
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from transformers.video_utils import VideoMetadata

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                                 device_map="cuda").eval()
    model.disable_talker()
    tok = processor.tokenizer
    yes_id, no_id = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    version = f"transformers {transformers.__version__} {MODEL} {TASK_VERSION}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = sorted(Path(args.videos_root).glob("*.mp4"))[args.shard::args.shards]
    if args.limit:
        videos = videos[:args.limit]
    t0 = time.time()
    for i, video in enumerate(videos, 1):
        target = out / f"{clip_id(video.stem)}.json"
        if target.exists():
            continue
        start = time.time()
        frames = [im for _, im in frames_per_second(video)]
        frames_all = list(frames)
        seconds = max(len(frames), 1)
        if len(frames) > MAX_FRAMES:
            keep = np.linspace(0, len(frames) - 1, MAX_FRAMES).round().astype(int)
            frames = [frames[k] for k in keep]
        audio = soundtrack(video)
        if args.timestamps:
            content, chunk_videos, chunk_audio, metas = chunked(frames_all, audio, args.timestamps, VideoMetadata)
            frames = [f for c in chunk_videos for f in c]
        else:
            chunk_videos, chunk_audio = [frames], [audio] if audio is not None else []
            content = [{"type": "video"}] + ([{"type": "audio"}] if audio is not None else [])
            fps = len(frames) / seconds
            metas = [VideoMetadata(total_num_frames=len(frames), fps=fps, duration=float(seconds),
                                   frames_indices=list(range(len(frames))))]
        content.append({"type": "text", "text": QUESTION})
        text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=text, videos=chunk_videos, audio=chunk_audio or None,
                           video_metadata=metas, do_sample_frames=False, return_tensors="pt")
        inputs = inputs.to(model.device)
        with torch.no_grad():
            logits = model.thinker(**inputs, logits_to_keep=1).logits[0, -1].float()
            p = torch.softmax(logits[[yes_id, no_id]], dim=-1)[0].item()
            gen = model.generate(**inputs, max_new_tokens=60, do_sample=False, return_audio=False)
        sequences = gen[0] if isinstance(gen, tuple) else gen
        text = processor.batch_decode(sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        decision, segments = parse_answer(text)
        record = {"clip_id": clip_id(video.stem), "model": version + (f" timestamps-{args.timestamps}s" if args.timestamps else ""),
                  "frames": len(frames),
                  "audio": audio is not None, "tokens": int(inputs["input_ids"].shape[1]),
                  "score": round(p, 4), "decision": decision, "segments": segments, "answer": text.strip(),
                  "seconds": round(time.time() - start, 1)}
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        tmp.rename(target)
        print(f"{i}/{len(videos)} {record['tokens']} tokens p={record['score']} {record['seconds']}s "
              f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
