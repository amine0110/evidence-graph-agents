"""Speech agent, step 3: tag threatening or distressed speech (GPU job).

For every transcript segment the speech agent keeps (speech.py), the reasoning
LLM reads the segment with the two segments before and after it and chooses
one of four labels. There is no generation: one forward pass per segment, and
the label probabilities are the softmax of the next-token logits over the four
label letters, so each tag carries a graded confidence.

  A threat    threatens violence, or demands something under threat of violence
  B distress  screams, begs for help or mercy, or expresses pain or terror
  C report    describes violence, a weapon, an injury, an accident or an explosion
  D none      none of these

Writes one JSON per clip (named by clip id) with the probabilities for each
speech node. Existing outputs are kept, so a stopped job resumes; --shard splits
the clips over several jobs. speech_tags.py turns the outputs into graph nodes.
"""

import argparse
import json
import time
from pathlib import Path

from speech import failed

MODEL = "google/gemma-4-12B-it"
# v1 defined a threat as "threatens someone, gives orders at weapon point, or
# verbally attacks someone"; the model then tagged ordinary orders and rude
# remarks (41% of normal clips with speech had a threat tag). v2 asks for
# threats of violence and excludes ordinary orders, arguments and surprise.
PROMPT_VERSION = "speech-tag-v2"
LABELS = {"A": "threat", "B": "distress", "C": "report", "D": "none"}
CONTEXT = 2
PROMPT = """You read speech transcribed automatically from a short video clip; the transcript may contain errors and may be in any language. Classify the TARGET line only; the lines around it are context.

A: threat of violence: the speaker threatens to hurt or kill someone, or demands something under threat of violence. Ordinary orders, arguments, insults and rude remarks are not threats.
B: distress: the speaker screams, begs for help or mercy, or expresses pain or terror. Ordinary surprise, sadness or annoyance is not distress.
C: report of violence: the speaker describes violence, a weapon, an injury, an accident or an explosion.
D: none of these.

Before:
{before}
TARGET [{start:.1f}-{end:.1f} s]: {text}
After:
{after}

Answer with one letter: A, B, C or D."""


def kept_segments(transcript):
    """The segments speech.py turns into nodes, with their node ids."""
    cid, out = transcript["clip_id"], []
    for seg in transcript["segments"]:
        if failed(seg) or not seg["text"]:
            continue
        out.append({"node": f"{cid}:speech:{len(out)}", **seg})
    return out


def prompts(segments):
    def lines(segs):
        return "\n".join(f"[{s['start']:.1f}-{s['end']:.1f} s] {s['text'].strip()}" for s in segs) or "(none)"
    for k, seg in enumerate(segments):
        yield PROMPT.format(before=lines(segments[max(0, k - CONTEXT):k]),
                            after=lines(segments[k + 1:k + 1 + CONTEXT]),
                            start=seg["start"], end=seg["end"], text=seg["text"].strip())


def score(model, tokenizer, label_ids, texts, batch=16):
    import torch
    rows = []
    for k in range(0, len(texts), batch):
        chats = [[{"role": "user", "content": t}] for t in texts[k:k + batch]]
        enc = tokenizer.apply_chat_template(chats, add_generation_prompt=True, tokenize=True, padding=True,
                                            return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        probs = torch.softmax(logits[:, label_ids].float(), dim=-1).tolist()
        rows += [{name: round(p, 4) for name, p in zip(LABELS.values(), row)} for row in probs]
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--asr", required=True, help="output folder of speech_asr.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="only the first N clips of this shard (tests)")
    args = parser.parse_args(argv)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"
    label_ids = []
    for letter in LABELS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        assert len(ids) == 1, (letter, ids)
        label_ids.append(ids[0])
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    version = f"transformers {transformers.__version__} {MODEL} {PROMPT_VERSION}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.asr).glob("x*.json"))[args.shard::args.shards]
    if args.limit:
        paths = paths[:args.limit]
    t0 = time.time()
    for i, path in enumerate(paths, 1):
        target = out / path.name
        if target.exists():
            continue
        transcript = json.loads(path.read_text(encoding="utf-8"))
        segments = kept_segments(transcript)
        probs = score(model, tokenizer, label_ids, list(prompts(segments))) if segments else []
        record = {"clip_id": transcript["clip_id"], "model": version, "language": transcript["language"],
                  "segments": [{"node": s["node"], "start": s["start"], "end": s["end"], "text": s["text"],
                                "probs": p} for s, p in zip(segments, probs)]}
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        tmp.rename(target)
        print(f"{i}/{len(paths)} {len(segments)} segments total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
