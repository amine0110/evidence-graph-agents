#!/bin/bash
# All GPU perception passes on the dev set (dev_set.py), with the same settings as the test runs:
# Whisper ASR, video detections + caption-v2 captions, speech-tag-v2 tags. Outputs under out/dev/.
# Submit from ~/c3:  sbatch code/experiments/c3_agents/slurm/dev_perception.sbatch
# The sound tagger runs on the Mac (audio_precheck.py scan), as it did for the test set.
#SBATCH --job-name=c3_dev_perc
#SBATCH --partition=rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Never pass --mem on this cluster: no memory is configured and the job is rejected.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
DEV=$PWD/data/XD-Violence/dev
OUT=$PWD/out/dev
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== asr"
PYLIB=$HOME/c3/pylib
PYTHONPATH=$PYLIB LD_LIBRARY_PATH=$PYLIB/nvidia/cublas/lib:$PYLIB/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-} \
    python3 code/experiments/c3_agents/speech_asr.py --videos-root "$DEV" --out "$OUT/asr"

export PYTHONPATH=$HOME/c3/vlib:$HOME/c3/code/src:$HOME/c3/code/experiments/c3_agents
export YOLO_CONFIG_DIR=$HOME/c3/.ultralytics YOLO_AUTOINSTALL=false PATH=/usr/local/cuda-12.8/bin:$PATH
echo "== video"
(cd models && python3 ../code/experiments/c3_agents/video_pass.py --videos-root "$DEV" --out "$OUT/video")

echo "== speech tags"
python3 code/experiments/c3_agents/speech_tag_pass.py --asr "$OUT/asr" --out "$OUT/speech_tags"
echo "done"
