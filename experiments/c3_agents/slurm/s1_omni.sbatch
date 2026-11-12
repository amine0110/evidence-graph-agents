#!/bin/bash
# S1 baseline: Qwen3-Omni-30B-A3B on the whole clip (about 62 GB in bf16).
# Submit from ~/c3:  SHARD=0 SHARDS=2 sbatch code/experiments/c3_agents/slurm/s1_omni.sbatch
#SBATCH --job-name=c3_s1_omni
#SBATCH --partition=rtxpro
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Never pass --mem on this cluster: no memory is configured and the job is rejected.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$HOME/c3/vlib:$HOME/c3/code/src:$HOME/c3/code/experiments/c3_agents
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 code/experiments/c3_agents/s1_omni.py --videos-root data/XD-Violence/test --out "out/${OUT:-s1_omni}" ${TS:+--timestamps "$TS"} \
    --shard "${SHARD:-0}" --shards "${SHARDS:-1}" ${LIMIT:+--limit "$LIMIT"}
