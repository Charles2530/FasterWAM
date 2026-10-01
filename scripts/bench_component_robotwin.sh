#!/usr/bin/env bash
# RoboTwin variant of the shared end-to-end contract (real text cached once).
set -euo pipefail
COMPONENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$COMPONENT_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
component_python="${BENCH_PYTHON:-python}"
component_checkpoint="${CKPT_PATH:-$COMPONENT_ROOT/checkpoints/fasterwam_release/robotwin/step_029355.pt}"
component_out="${OUT:-$COMPONENT_ROOT/artifacts/robotwin_components_$(date +%Y%m%d_%H%M%S)}"
"$component_python" scripts/bench_component.py --task robotwin --checkpoint "$component_checkpoint" \
    --warmup 10 --iterations 100 --steps 10 1 --output-dir "$component_out"
