"""Speech agent, step 2: turn Whisper transcripts into graph nodes.

One transcript_segment node per Whisper segment. Confidence is the segment's
mean token probability times the probability that it holds speech. Segments
Whisper itself would treat as failed (compression ratio above 2.4, a sign of
a repetition loop, or no-speech probability above 0.6 with mean log
probability below -1) are dropped and counted.

The threat and distress tags that the reasoning LLM adds later go in their
own agent folder, linked to these nodes by derived_from edges.
"""

import argparse
import json
import math
from pathlib import Path

from evgraph import EvidenceGraph, Modality, Node, Source

PRODUCER = "speech"
MAX_COMPRESSION, NO_SPEECH, LOW_LOGPROB = 2.4, 0.6, -1.0


def failed(seg):
    return seg["compression_ratio"] > MAX_COMPRESSION or (
        seg["no_speech_prob"] > NO_SPEECH and seg["avg_logprob"] < LOW_LOGPROB)


def speech_nodes(transcript):
    cid, nodes = transcript["clip_id"], []
    for seg in transcript["segments"]:
        if failed(seg) or not seg["text"]:
            continue
        confidence = math.exp(min(seg["avg_logprob"], 0.0)) * (1 - seg["no_speech_prob"])
        nodes.append(Node(
            id=f"{cid}:speech:{len(nodes)}",
            modality=Modality.SPEECH,
            kind="transcript_segment",
            content={"text": seg["text"], "language": transcript["language"]},
            t_obs=(seg["start"], seg["end"]),
            t_rec=seg["end"],
            source=Source(producer=PRODUCER, version=transcript["model"], input_ref=cid,
                          offset=f"{seg['start']:.1f}-{seg['end']:.1f}s"),
            confidence=round(min(max(confidence, 0.0), 1.0), 4),
        ))
    return nodes


def build(asr_dir, graphs_root):
    out_dir = Path(graphs_root) / PRODUCER
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = nodes = dropped = 0
    languages = {}
    for path in sorted(Path(asr_dir).glob("x*.json")):
        transcript = json.loads(path.read_text(encoding="utf-8"))
        target = out_dir / f"{transcript['clip_id']}.jsonl"
        target.write_text("")
        graph = EvidenceGraph(target)
        made = speech_nodes(transcript)
        for node in made:
            graph.add(node)
        clips += 1
        nodes += len(made)
        dropped += sum(failed(s) for s in transcript["segments"])
        # With no speech left after the voice filter, Whisper still guesses a language
        # (often "nn"); count those clips as having no speech instead.
        language = transcript["language"] if made else "none"
        languages[language] = languages.get(language, 0) + 1
    return {"clips": clips, "nodes": nodes, "dropped_segments": dropped,
            "languages": dict(sorted(languages.items(), key=lambda kv: -kv[1]))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--asr", required=True, help="output folder of speech_asr.py")
    parser.add_argument("--out", required=True, help="graphs root; this agent writes <out>/speech/")
    args = parser.parse_args(argv)
    print(build(args.asr, args.out))


if __name__ == "__main__":
    main()
