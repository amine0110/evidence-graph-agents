"""Video agent, step 1: detections and captions for each clip (GPU job).

Decodes one frame per second and runs, per clip:
  detect   two detectors on every frame: a YOLOv8-L trained earlier in this
           thesis on people and weapons (person, grenade, knife, machine gun,
           pistol, RPG), and an open-vocabulary detector (YOLO-World) for the
           cues that model does not cover (fire, smoke, vehicles, police).
  caption  a vision-language model on 4-second chunks (4 frames each), asked
           for a short factual description of what happens.

Writes one JSON per clip, named by its neutral clip id. Existing outputs are
kept, so a stopped job resumes; --shard splits the clips over several jobs.
Step 2 (video.py) turns the outputs into graph nodes.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

WEAPONS = "person_weapons.pt"   # earlier work, 2024; classes are read from the weights
DETECTOR = "yolov8x-worldv2.pt"
OBJECTS = ["fire", "smoke", "car", "motorcycle", "police officer", "stick"]
DETECT_MIN = 0.15
VLM = "Qwen/Qwen3-VL-8B-Instruct"
CHUNK = 4  # seconds per caption, one frame per second
FRAME_SIDE = 448
PROMPT = ("These are {n} frames, one second apart, from a video. In at most 40 words, "
          "describe factually what the people do and what happens. Mention weapons, fire, "
          "vehicles or injuries only if you see them.")


def clip_id(stem):
    return "x" + hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]


def frames_per_second(path):
    """(second, PIL image) for the first frame at or after each whole second."""
    import av
    out, next_t = [], 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None or frame.time + 1e-6 < next_t:
                continue
            image = frame.to_image()
            image.thumbnail((FRAME_SIDE, FRAME_SIDE))
            out.append((int(round(next_t)), image))
            next_t += 1.0
    return out


def detect(detector, name, frames):
    rows = []
    for k in range(0, len(frames), 16):
        batch = frames[k:k + 16]
        results = detector.predict([im for _, im in batch], conf=DETECT_MIN, verbose=False)
        for (second, _), result in zip(batch, results):
            boxes = result.boxes
            for cls, conf, box in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxyn.tolist()):
                rows.append({"t": second, "model": name, "object": result.names[int(cls)],
                             "conf": round(conf, 4), "box": [round(x, 3) for x in box]})
    return rows


def caption(model, processor, frames, batch=8):
    import torch
    chunks = [frames[k:k + CHUNK] for k in range(0, len(frames), CHUNK)]
    rows = []
    for k in range(0, len(chunks), batch):
        group = chunks[k:k + batch]
        messages = [[{"role": "user", "content": [*({"type": "image", "image": im} for _, im in chunk),
                                                  {"type": "text", "text": PROMPT.format(n=len(chunk))}]}]
                    for chunk in group]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt",
                                               processor_kwargs={"padding": True})
        inputs = inputs.to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for chunk, text in zip(group, texts):
            rows.append({"start": chunk[0][0], "end": chunk[-1][0] + 1, "text": text.strip()})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="only the first N clips of this shard (tests)")
    args = parser.parse_args(argv)

    import torch
    import transformers
    import ultralytics
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from ultralytics import YOLO, YOLOWorld

    weapons = YOLO(WEAPONS)
    detector = YOLOWorld(DETECTOR)
    detector.set_classes(OBJECTS)
    processor = AutoProcessor.from_pretrained(VLM)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(VLM, dtype=torch.bfloat16, device_map="cuda").eval()
    versions = {"detectors": {WEAPONS: f"ultralytics {ultralytics.__version__} {WEAPONS}",
                              DETECTOR: f"ultralytics {ultralytics.__version__} {DETECTOR}"},
                "vlm": f"transformers {transformers.__version__} {VLM}"}

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
        frames = frames_per_second(video)
        record = {"clip_id": clip_id(video.stem), **versions, "frames": len(frames),
                  "detections": detect(weapons, WEAPONS, frames) + detect(detector, DETECTOR, frames),
                  "captions": caption(model, processor, frames)}
        record["seconds"] = round(time.time() - start, 1)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        tmp.rename(target)
        print(f"{i}/{len(videos)} {record['frames']} frames {record['seconds']}s total {time.time() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
