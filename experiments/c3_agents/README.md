# c3_agents

Perception agents, edge builder, the four compared systems and the evaluation for the paper. The steps are listed under "Pipeline" below.

## Audio pre-check

`audio_precheck.py` measures loudness and runs an AudioSet tagger (AST) on
every XD-Violence test clip, then sorts clips into speech, other sound, music
only and near-silent, checks for a silence shortcut across classes, and
compares the tagger with the annotators' sound and speech ticks. Needs
`torch`, `transformers`, `av` and `scipy`.

```bash
python experiments/c3_agents/audio_precheck.py scan --videos-root ~/data/XD-Violence/test --out ~/data/XD-Violence/precheck
python experiments/c3_agents/audio_precheck.py report --out ~/data/XD-Violence/precheck \
    --annotator ~/data/XD-Violence/amine --annotator ~/data/XD-Violence/annotator-b
```

The report reads the annotations with the `annotator` package from the
video-review-annotator repo, expected next to this one.

## Pipeline (Domain A, XD-Violence test set)

GPU steps run on the lab SLURM cluster from `~/c3` (job scripts in `slurm/`); graph building
and evaluation run on the Mac, which alone holds `graphs/clips.csv` (clip id to file name and
label). The cluster copy of the graphs never contains `clips.csv`.

| Step | Script | Writes |
|---|---|---|
| Sound events | `audio_precheck.py scan`, then `audio_events.py` | `graphs/audio_events/` |
| Speech | `speech_asr.py` (GPU), then `speech.py` | `graphs/speech/` |
| Speech tags | `speech_tag_pass.py` (GPU, prompt speech-tag-v2), then `speech_tags.py` | `graphs/speech_tags/` |
| Video | `video_pass.py` (GPU, detectors + caption-v2), then `video.py` | `graphs/video/` |
| Edges | `edges.py` (rules: concept lexicon + time overlap) | `graphs/edges/` |
| Systems | `reasoning.py` (S2, S3, S3 v2, S4, served by vLLM), `s1_omni.py` (S1, S1-ts) | run files |
| Scores | `evaluate.py` | `frozen/eval_final.json` |

All systems answer the question in `task.py` and are scored by P(Yes) on the first answer token.
`frozen/` holds the evaluation set hash (`eval_set.json`, all 799 clips), the dev set used to
develop S3 v2 (`dev_set.json`, 152 training clips, built by `dev_set.py`), the final scores and
the cost per clip. S3 v2 was changed on the dev set only; the protocol and each attempt are
described in the paper (development set section).
