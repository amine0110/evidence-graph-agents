# c3_agents

Design: see the matching page under "Experiment design notes C1 to C3 (2026-10-28)" in Notion (PhD Get Together). Code for this experiment goes here.

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
