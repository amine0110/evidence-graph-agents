#!/bin/bash
# Threat and distress tagging of the transcripts (speech agent, step 3).
# Submit from ~/c3:  SHARD=0 SHARDS=4 sbatch code/experiments/c3_agents/slurm/speech_tag.sbatch
# Gemma 4 12B in bf16 needs about 26 GB: RTX PRO 6000 or 4500, or a 3090/4090 is too small.
#SBATCH --job-name=c3_speech_tag
#SBATCH --partition=rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Never pass --mem on this cluster: no memory is configured and the job is rejected.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$HOME/c3/vlib:$HOME/c3/code/src:$HOME/c3/code/experiments/c3_agents
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 code/experiments/c3_agents/speech_tag_pass.py --asr out/asr --out out/speech_tags \
    --shard "${SHARD:-0}" --shards "${SHARDS:-1}" ${LIMIT:+--limit "$LIMIT"}
