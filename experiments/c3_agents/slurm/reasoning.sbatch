#!/bin/bash
# S2 / S3 / S4 and ablations: a vLLM server for the reasoning LLM plus the client, in one job.
# Submit from ~/c3, e.g.:
#   RUNS="s3:--system s3;s4:--system s4" sbatch code/experiments/c3_agents/slurm/reasoning.sbatch
# RUNS lists runs separated by ';', each "name:arguments to reasoning.py"; outputs go to
# out/reasoning/<name>.jsonl, and a rerun skips clips already there.
#SBATCH --job-name=c3_reason
#SBATCH --partition=rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Never pass --mem on this cluster: no memory is configured and the job is rejected.
set -euo pipefail
# pip --target does not run .pth files, so the CUTLASS DSL folder (needed by Qwen3.6) is added by hand
cd "$SLURM_SUBMIT_DIR"
MODEL=${MODEL:-google/gemma-4-12B-it}
PORT=$((8000 + SLURM_JOB_ID % 1000))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PYTHONPATH=$HOME/c3/vllmlib:$HOME/c3/vllmlib/nvidia_cutlass_dsl/dsl_packages PATH=$HOME/c3/vllmlib/bin:$PATH \
    python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" \
    --max-model-len "${MAX_LEN:-65536}" --gpu-memory-utilization 0.90 ${MAX_SEQS:+--max-num-seqs "$MAX_SEQS"} --tensor-parallel-size "${TP:-1}" \
    > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
    kill -0 $SERVER 2>/dev/null || { echo "vLLM server died"; tail -30 "logs/vllm_${SLURM_JOB_ID}.log"; exit 1; }
    sleep 10
done
echo "server up"
export PYTHONPATH=$HOME/c3/vlib:$HOME/c3/code/src:$HOME/c3/code/experiments/c3_agents
mkdir -p out/reasoning
IFS=';' read -ra LIST <<< "$RUNS"
for run in "${LIST[@]}"; do
    name=${run%%:*}; rest=${run#*:}
    echo "== $name: $rest"
    # shellcheck disable=SC2086
    python3 code/experiments/c3_agents/reasoning.py --graphs "${GRAPHS:-graphs}" --server "http://localhost:$PORT" \
        --model "$MODEL" --out "out/reasoning/$name.jsonl" $rest
done
