#!/usr/bin/env bash
# Native 10-step / 1-step components, two independent real-text processes.
set -euo pipefail
COMPONENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$COMPONENT_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CKPT_PATH="${CKPT_PATH:-/mnt/miaohua/charles/models/fasterwam_release/robotwin/step_029355.pt}"
export DATASET_STATS_PATH="${DATASET_STATS_PATH:-/mnt/miaohua/charles/datasets/FastWAM-RoboTwin/dataset_stats.json}"
export DIFFSYNTH_MODEL_BASE_PATH="$COMPONENT_ROOT/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
component_out="${OUT:-$COMPONENT_ROOT/artifacts/robotwin_components_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$component_out")"
mkdir "$component_out"
component_gpu_pid=""
trap 'if [[ -n "$component_gpu_pid" ]]; then kill "$component_gpu_pid" 2>/dev/null || true; wait "$component_gpu_pid" 2>/dev/null || true; fi' EXIT
for text_mode in per_request cached; do
    # A compute process can be idle between requests; require free memory as well.
    nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$component_out/$text_mode.gpu_before.csv"
    component_mem="$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)"
    component_util="$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
    if (( component_mem > 512 || component_util > 0 )); then
        echo "GPU $CUDA_VISIBLE_DEVICES is occupied ($component_mem MiB, $component_util%); select a free GPU_ID and rerun with a new OUT." >&2
        exit 1
    fi
    nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm --format=csv --loop-ms=1000 > "$component_out/$text_mode.gpu_telemetry.csv" &
    component_gpu_pid=$!
    .venvs/robotwin/bin/python scripts/bench_component.py --text-mode "$text_mode" \
        --warmup 100 --iters 100 --steps 10 1 --output-dir "$component_out/$text_mode" \
        > "$component_out/$text_mode.log" 2>&1
    kill "$component_gpu_pid" 2>/dev/null || true
    wait "$component_gpu_pid" 2>/dev/null || true
    component_gpu_pid=""
    cat "$component_out/$text_mode/components.md"
done
echo "Saved all samples and telemetry: $component_out"
