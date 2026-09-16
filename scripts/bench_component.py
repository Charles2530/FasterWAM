"""Native RoboTwin FasterWAM component timing (10 and 1 action steps).

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CKPT_PATH=/path/to/step_029355.pt DATASET_STATS_PATH=/path/to/dataset_stats.json \
  .venvs/robotwin/bin/python scripts/bench_component.py --text-mode per_request

Use --text-mode cached in a separate process to cache the real prompt once.
Real dataset observations are resized/normalized outside timing. Each request
includes their CPU-to-GPU copies, noise generation, VAE, video preparation and
one-pass future KV cache, action denoising, scheduler, and CPU action output.
No model/operator changes, CUDA Graph, compilation, or simulator are used.

CUDA events measure intervals in the native call without component barriers.
other = profiled request wall time - union of marked CUDA intervals. It includes
unmarked operations, host submission gaps and synchronization, not just CPU work.
Use the separately measured uninstrumented total for performance comparisons.
All samples are retained. Loading, preprocessing, warmups and validation are
excluded from formal statistics. GPU telemetry is recorded outside timed calls.
"""

from __future__ import annotations

import os

# Set before importing torch/BLAS, including for direct script invocations.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import av
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from experiments.robotwin.fasterwam_policy.deploy_policy import WorldActionRobotWinPolicy
from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

COMPONENTS = ("vae_encode", "text_encode", "video_prepare", "video_dit_prefill",
              "action_dit_denoise", "action_scheduler")
ROWS = (*COMPONENTS, "overlap_correction", "other", "total", "total_uninstrumented")


def interval_accounting(intervals, wall_ms):
    values = dict.fromkeys(COMPONENTS, 0.0)
    for name, start, end in intervals:
        if end < start:
            raise ValueError(f"Negative interval for {name}")
        values[name] += end - start
    union, right = 0.0, float("-inf")
    for start, end in sorted((s, e) for _, s, e in intervals):
        union += max(0.0, end - max(start, right))
        right = max(right, end)
    return dict(values, overlap_correction=union - sum(values.values()),
                other=wall_ms - union, total=wall_ms)


def summary(samples):
    ordered = sorted(samples)
    def percentile(p):
        pos = (len(ordered) - 1) * p / 100
        low = int(pos)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (pos - low)
    return dict(mean_ms=statistics.fmean(samples), p50_ms=percentile(50), p90_ms=percentile(90),
                min_ms=min(samples), max_ms=max(samples))


def snapshot():
    cgroup = Path("/sys/fs/cgroup")
    result = {"utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
              "intra_op_threads": torch.get_num_threads(),
              "inter_op_threads": torch.get_num_interop_threads(),
              "os_threads": len(list(Path("/proc/self/task").iterdir()))}
    for name in ("cpu.max", "cpu.stat", "cpuset.cpus.effective"):
        path = cgroup / name
        result[name] = path.read_text() if path.exists() else None
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,index,uuid,name,utilization.gpu,"
                          "memory.used,power.draw,temperature.gpu,clocks.sm", "--format=csv"],
                         capture_output=True, text=True, timeout=15)
    result["gpu_telemetry"] = gpu.stdout if gpu.returncode == 0 else gpu.stderr
    return result


class ComponentTimer:
    def __init__(self):
        self.origin = torch.cuda.Event(enable_timing=True)
        self.pool = {}
        self.reset()

    def reset(self):
        self.counts, self.used = {}, []

    def wrap(self, name, function):
        @wraps(function)
        def timed(*args, **kwargs):
            index = self.counts.get(name, 0)
            self.counts[name] = index + 1
            key = name, index
            if key not in self.pool:
                self.pool[key] = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start, end = self.pool[key]
            start.record()
            value = function(*args, **kwargs)
            end.record()
            self.used.append((name, start, end))
            return value
        return timed

    def values(self, wall_ms):
        return interval_accounting([(name, self.origin.elapsed_time(start), self.origin.elapsed_time(end))
                                    for name, start, end in self.used], wall_ms)


@contextmanager
def instrument(model, timer):
    hooks = [(model, "_encode_input_image_latents_tensor", "vae_encode"),
             (model, "encode_prompt", "text_encode"),
             (model.video_expert, "pre_dit", "video_prepare"),
             (model.mot, "prefill_video_cache", "video_dit_prefill"),
             (model, "_predict_action_noise_with_cache", "action_dit_denoise"),
             (model.infer_action_scheduler, "step", "action_scheduler")]
    originals = []
    try:
        for obj, attr, name in hooks:
            owned, original = attr in vars(obj), getattr(obj, attr)
            originals.append((obj, attr, owned, original))
            setattr(obj, attr, timer.wrap(name, original))
        yield
    finally:
        for obj, attr, owned, original in reversed(originals):
            if owned:
                setattr(obj, attr, original)
            else:
                delattr(obj, attr)


def load_observations(args, cfg):
    """Use deployment's exact mosaic and state normalization on real frames."""
    root = args.dataset_root
    info = json.loads((root / "meta/info.json").read_text())
    keys = dict(episode_chunk=args.episode // info["chunks_size"], episode_index=args.episode)
    table = pq.read_table(root / info["data_path"].format(**keys),
                          columns=["observation.state", "task_index"])
    indices = [0, min(50, len(table) - 1)]
    rows = table.to_pylist()
    task_index = rows[indices[0]]["task_index"]
    with (root / "meta/tasks.jsonl").open() as file:
        task = next(row["task"] for line in file if (row := json.loads(line))["task_index"] == task_index)
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.processor = instantiate(cfg.data.train.processor).eval()
    policy.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(args.dataset_stats)))
    # Keep prepared observations on CPU: native infer includes H2D copies.
    policy.model = SimpleNamespace(device="cpu", torch_dtype=torch.bfloat16)
    observations = [{"observation": {}} for _ in indices]
    paths = []
    for camera, key in (("head_camera", "cam_high"), ("left_camera", "cam_left_wrist"),
                        ("right_camera", "cam_right_wrist")):
        path = root / info["video_path"].format(**keys, video_key=f"observation.images.{key}")
        paths.append(str(path))
        with av.open(str(path)) as container:
            container.streams.video[0].thread_count = 1
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index in indices:
                    rgb = frame.to_ndarray(format="rgb24")
                    for i, wanted in enumerate(indices):
                        if wanted == frame_index:
                            observations[i]["observation"][camera] = {"rgb": rgb}
                if frame_index >= max(indices):
                    break
    cases = [(policy._build_robotwin_image_tensor(obs),
              policy._normalize_state(np.asarray(rows[idx]["observation.state"], dtype=np.float32)), args.seed)
             for obs, idx in zip(observations, indices)]
    cases.append((*cases[0][:2], args.seed + 1))
    metadata = dict(episode=args.episode, frame_indices=indices, task_index=task_index,
                    task=task, videos=paths, image_shape=list(cases[0][0].shape),
                    proprio_shape=list(cases[0][1].shape), source="real_dataset_observations")
    return cases, DEFAULT_PROMPT.format(task=task), metadata


def run_pass(request, args, timer=None, expected_counts=None):
    before = snapshot()
    for _ in range(args.warmup):
        if timer:
            timer.reset()
        request()
        torch.cuda.synchronize()
    after_warmup = snapshot()
    samples, components = [], []
    for index in range(args.iters):
        torch.cuda.synchronize()
        if timer:
            timer.reset()
        start = time.perf_counter()
        if timer:
            timer.origin.record()
        request()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        samples.append(elapsed)
        if timer:
            if timer.counts != expected_counts:
                raise RuntimeError(f"Unexpected component calls: {timer.counts}, expected {expected_counts}")
            components.append(timer.values(elapsed))
        if (index + 1) % 25 == 0:
            print(f"  {'profiled' if timer else 'clean'} sample {index+1}/{args.iters}: {elapsed:.3f} ms", flush=True)
    return dict(samples_ms=samples, summary=summary(samples), component_samples_ms=components,
                monitoring=dict(before=before, after_warmup=after_warmup, after=snapshot()))


def save_report(report, output_dir):
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    results = report["results"]
    lines = [f"FasterWAM RoboTwin native components; text mode: {report['text_mode']}. All times in ms.", "",
             "| Component | " + " | ".join(f"{r['steps']} steps" for r in results) + " |",
             "| --- | " + " | ".join("---:" for _ in results) + " |"]
    for row in ROWS:
        lines.append(f"| {row} | " + " | ".join(f"{r['components_ms'][row]:.3f}" for r in results) + " |")
    lines += ["", "Components are CUDA intervals in the instrumented native request. "
              "other is wall time minus interval union, including unmarked work and synchronization. "
              "Use total_uninstrumented to compare latency; all totals include VAE, input transfers, "
              "noise and CPU output. Video prefill includes future-frame KV fusion. "
              "Action denoise includes pre_dit, all action blocks and post_dit. "
              "Dataset decode, mosaic/state normalization, model load and validation are excluded.", "",
              "| Steps | Mean | p50 | p90 | Min | Max |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        s = r["clean"]["summary"]
        lines.append(f"| {r['steps']} | " + " | ".join(f"{s[k]:.3f}" for k in
                      ("mean_ms", "p50_ms", "p90_ms", "min_ms", "max_ms")) + " |")
    (output_dir / "components.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("CKPT_PATH"), required="CKPT_PATH" not in os.environ)
    parser.add_argument("--dataset-stats", type=Path, default=os.environ.get("DATASET_STATS_PATH"), required="DATASET_STATS_PATH" not in os.environ)
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/miaohua/charles/datasets/FastWAM-RoboTwin/robotwin2.0"))
    parser.add_argument("--text-mode", choices=["per_request", "cached"], default="per_request")
    parser.add_argument("--steps", type=int, nargs="+", choices=[10, 1], default=[10, 1])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / f"bench_component_{datetime.now():%Y%m%d_%H%M%S}")
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0 or len(set(args.steps)) != len(args.steps):
        parser.error("Require warmup>=0, iters>0 and distinct step counts")
    for path in (args.checkpoint, args.dataset_stats, args.dataset_root):
        if not path.exists():
            parser.error(f"Missing required path: {path}")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_num_threads(1)
    assert torch.get_num_threads() == 1
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    initial = snapshot()
    (args.output_dir / "initial_system.json").write_text(json.dumps(initial, indent=2) + "\n")
    print(f"Native benchmark: torch={torch.__version__}, intra_op={torch.get_num_threads()}, "
          f"inter_op={torch.get_num_interop_threads()}, GPU={torch.cuda.get_device_name()}", flush=True)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="sim_robotwin", overrides=["EVALUATION.sigma_shift=5.0"])
    cases, prompt, input_metadata = load_observations(args, cfg)
    torch.save(dict(cases=cases, prompt=prompt), args.output_dir / "inputs.pt")
    print("Loading RoboTwin model and checkpoint...", flush=True)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0").eval()
    payload = model.load_checkpoint(str(args.checkpoint))
    checkpoint_step = payload.get("step")
    del payload
    context, mask = model.encode_prompt(prompt)
    report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_step=checkpoint_step,
                  dataset_stats=str(args.dataset_stats.resolve()),
                  dataset_stats_sha256=hashlib.sha256(args.dataset_stats.read_bytes()).hexdigest(),
                  torch=torch.__version__, python=sys.executable, gpu=torch.cuda.get_device_name(),
                  dtype="torch.bfloat16", cpu_threads=torch.get_num_threads(),
                  cpu_interop_threads=torch.get_num_interop_threads(), text_mode=args.text_mode,
                  inference_path="infer_action_one_pass_future_cache", action_horizon=32,
                  num_video_frames=9, sigma_shift=5.0, rand_device="cpu", verify=True,
                  warmup=args.warmup, iterations=args.iters, input=input_metadata, prompt=prompt,
                  config=OmegaConf.to_container(cfg, resolve=False), initial_system=initial,
                  command=shlex.join(["env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                                     f"CKPT_PATH={args.checkpoint.resolve()}",
                                     f"DATASET_STATS_PATH={args.dataset_stats.resolve()}",
                                     *[f"{key}={os.environ[key]}" for key in
                                       ("CUDA_VISIBLE_DEVICES", "DIFFSYNTH_MODEL_BASE_PATH",
                                        "HF_HUB_OFFLINE", "DIFFSYNTH_SKIP_DOWNLOAD") if key in os.environ],
                                     sys.executable, *sys.argv]), results=[])
    (args.output_dir / "command.txt").write_text(
        report["command"] + "\n")
    infer = model.infer_action_one_pass_future_cache
    common = dict(action_horizon=32, num_video_frames=9, sigma_shift=5.0, rand_device="cpu", tiled=False)
    for steps in args.steps:
        print(f"Benchmarking {steps} steps, text={args.text_mode}", flush=True)
        def request(case=cases[0], cached=args.text_mode == "cached"):
            image, proprio, seed = case
            return infer(prompt=None if cached else prompt, context=context if cached else None,
                         context_mask=mask if cached else None, input_image=image, proprio=proprio,
                         seed=seed, num_inference_steps=steps, **common)["action"]
        references = [request(case, cached=True) for case in cases]
        if torch.equal(references[0], references[1]) or torch.equal(references[0], references[2]):
            raise RuntimeError("Validation inputs do not change the action")
        for case, reference in zip(cases, references):
            torch.testing.assert_close(request(case), reference, atol=0, rtol=0)
        clean = run_pass(request, args)
        timer = ComponentTimer()
        errors = []
        with instrument(model, timer):
            for case, reference in zip(cases, references):
                timer.reset()
                actual = request(case)
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                errors.append((actual - reference).abs().max().item())
            expected = dict(vae_encode=1, video_prepare=1, video_dit_prefill=1,
                            action_dit_denoise=steps, action_scheduler=steps)
            if args.text_mode == "per_request":
                expected["text_encode"] = 1
            profiled = run_pass(request, args, timer, expected)
        rows = profiled["component_samples_ms"]
        components = {k: statistics.fmean(row[k] for row in rows) for k in rows[0]}
        components["total_uninstrumented"] = clean["summary"]["mean_ms"]
        report["results"].append(dict(steps=steps, components_ms=components, clean=clean, profiled=profiled,
                                      validation_max_abs_errors=errors, component_calls=expected,
                                      profiling_overhead_ms=components["total"] - components["total_uninstrumented"]))
        save_report(report, args.output_dir)
        print(components, flush=True)
    print((args.output_dir / "components.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
