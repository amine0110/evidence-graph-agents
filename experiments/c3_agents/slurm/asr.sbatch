#!/bin/bash
# Whisper transcription of the XD-Violence test clips (speech agent, step 1).
# Submit from ~/c3:  sbatch code/experiments/c3_agents/slurm/asr.sbatch
#SBATCH --job-name=c3_asr
#SBATCH --partition=rtx40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Never pass --mem on this cluster: no memory is configured and the job is rejected.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYLIB=$HOME/c3/pylib
export PYTHONPATH=$PYLIB
export LD_LIBRARY_PATH=$PYLIB/nvidia/cublas/lib:$PYLIB/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 code/experiments/c3_agents/speech_asr.py \
    --videos-root data/XD-Violence/test --out out/asr \
    --shard "${SHARD:-0}" --shards "${SHARDS:-1}"
