#!/bin/bash
# Detections and captions for the XD-Violence test clips (video agent, step 1).
# Submit from ~/c3, one job per shard, for example:
#   for s in 0 1 2 3; do SHARD=$s SHARDS=4 sbatch code/experiments/c3_agents/slurm/video.sbatch; done
#SBATCH --job-name=c3_video
#SBATCH --partition=rtx40,rtx30,rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Never pass --mem on this cluster: no memory is configured and the job is rejected.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$HOME/c3/vlib
export PATH=/usr/local/cuda-12.8/bin:$PATH
export YOLO_CONFIG_DIR=$HOME/c3/.ultralytics
export YOLO_AUTOINSTALL=false   # never let ultralytics pip-install into the user site
mkdir -p models && cd models   # person_weapons.pt is copied here; YOLO-World downloads here
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 ../code/experiments/c3_agents/video_pass.py \
    --videos-root ../data/XD-Violence/test --out ../out/video \
    --shard "${SHARD:-0}" --shards "${SHARDS:-1}" ${LIMIT:+--limit $LIMIT}
