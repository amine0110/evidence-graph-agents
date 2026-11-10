# evidence-graph-agents

Experiment code for the PhD thesis *knowledge-graph-enhanced agents for
multimodal reasoning and decision support*. Perception agents write typed,
time-stamped observations into a shared evidence graph; a reasoning agent reads
it. Paper sources live in their own repositories, not here.

## Layout

| Path | Content | Thesis |
| --- | --- | --- |
| `src/evgraph/` | evidence graph schema v1 and append-only store, shared by all experiments | Ch 3 |
| `experiments/c1_relation/` | C1: joint similarity and stance relation model on DisStance | Ch 5 |
| `experiments/c2_reasoning/` | C2: stance- and time-aware context selection, belief revision | Ch 6 |
| `experiments/c3_agents/` | C3: perception agents and reasoning agent on security, disinformation and drone data | Ch 7 |

The experiment design (hypotheses, baselines, controls, decision rules) is
frozen in the Notion page "Experiment design notes C1 to C3 (2026-10-28)"
under PhD Get Together. Change the design there first, with a dated entry in
its change log.

## Evidence graph schema v1

- **Node** = one observation, never a world entity: `id`, `modality` (text,
  image, video, audio_event, speech), `kind`, `content`, `t_obs` (interval
  observed), `t_rec` (when recorded), `source` (producer, version, input,
  offset), `confidence`, optional ground-truth `label`.
- **Edge** = typed relation: `same_event` (symmetric, graded), `supports`,
  `contradicts` (directed, graded), `precedes`, `derived_from` (directed, no
  score). Every edge has a producer and a confidence.
- **Veracity** = a belief value for a node at a time, produced by a revision
  rule. Beliefs are appended, never overwritten; the newest counts.
- Storage: one JSONL file per graph, one record per line.

```python
from evgraph import EvidenceGraph, Node, Edge, EdgeType, Modality, Source

g = EvidenceGraph("graph.jsonl")
past = g.neighbours("query", types=[EdgeType.CONTRADICTS], recorded_before=query.t_rec)
```

## Rules for runs

- GPU runs happen on the lab SLURM servers (since 2026-10-30; no longer the
  Z790). Record the model file, quantisation and runtime version for every run.
- Every results file records the code commit and the data split hash.
- Fixed seeds; temperature 0 for LLM inference; three seeds for training.
- Report negative results.

## Setup and tests

```bash
pip install -e ".[test]"
pytest
```
