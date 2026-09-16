#!/usr/bin/env bash
# Local model, dataset, and rendering paths for this machine.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export DIFFSYNTH_MODEL_BASE_PATH="${REPO_ROOT}/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUROBO_TORCH_COMPILE_DISABLE=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

# Reuse the NVIDIA rendering libraries already configured for RoboTwin.
ROBOTWIN_RENDER_ENV=/mnt/miaohua/charles/envs/miniconda3/envs/RoboTwin
export OIDN_ROOT="${ROBOTWIN_RENDER_ENV}/opt/oidn-2.4.1.x86_64.linux"
export ROBOTWIN_NVIDIA_GL_ROOT="${ROBOTWIN_RENDER_ENV}/nvidia_gl_550_extracted"
export LD_LIBRARY_PATH="${ROBOTWIN_NVIDIA_GL_ROOT}:${OIDN_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VK_ICD_FILENAMES="${ROBOTWIN_NVIDIA_GL_ROOT}/nvidia_icd_abs.json"
export LD_PRELOAD="${ROBOTWIN_NVIDIA_GL_ROOT}/libGL.so.1.7.0${LD_PRELOAD:+:${LD_PRELOAD}}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${ROBOTWIN_NVIDIA_GL_ROOT}/10_nvidia.json"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export XDG_RUNTIME_DIR="${REPO_ROOT}/.runtime/robotwin"
unset ROBOTWIN_FORCE_RASTER LIBGL_ALWAYS_SOFTWARE MESA_LOADER_DRIVER_OVERRIDE
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

export TASK_NAME=robotwin_fasterwam_3cam_384_1e-4
export CKPT_PATH="${CKPT_PATH:-/mnt/miaohua/charles/models/fasterwam_release/robotwin/step_029355.pt}"
export DATASET_STATS_PATH="${DATASET_STATS_PATH:-/mnt/miaohua/charles/models/fasterwam_release/robotwin/dataset_stats.json}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
export REPLAN_STEPS="${REPLAN_STEPS:-28}"

# The upstream manager uses the final output-directory component as the run ID.
OUT="${OUT:-${REPO_ROOT}/evaluate_results/robotwin/step_029355/fasterwam_10step_$(date +%Y%m%d_%H%M%S)}"
exec bash scripts/eval_fasterwam_robotwin.sh \
  EVALUATION.eval_num_episodes=100 \
  EVALUATION.num_inference_steps=10 \
  EVALUATION.action_infer_mode=one_pass_future_cache \
  EVALUATION.sigma_shift=5.0 \
  EVALUATION.instruction_type=unseen \
  EVALUATION.skip_get_obs_within_replan=true \
  EVALUATION.timing_enabled=true \
  "EVALUATION.output_dir=${OUT}" \
  "$@"
