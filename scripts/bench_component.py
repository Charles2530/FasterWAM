"""FasterWAM component timing with the same request contract as the 13-group table.

Matches bench_latency.py: real cached text, synthetic CPU observations, CPU RNG,
input copies, VAE, proprio, preparation, Video KV prefill, 10/1 action steps and
CPU output; no_grad, one CPU thread, synchronized end-to-end wall time.
Default task/checkpoint/seed/warmup/iterations are shared with
infer_video_action_random.py. Component events are a separate profiling pass;
use total_uninstrumented for comparison with its groups 1 and 2.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import os
import time

from fasterwam_benchmark import (
    BenchmarkSetup, ROOT, add_common_arguments, benchmark, check_common_arguments,
    summarize as summary, torch,
)

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument('--steps', type=int, nargs='+', choices=[10, 1], default=[10, 1])
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / f'artifacts/fasterwam_components_{datetime.now():%Y%m%d_%H%M%S}')
    args = check_common_arguments(parser, parser.parse_args())
    if len(set(args.steps)) != len(args.steps):
        parser.error('steps must be distinct')
    return args


def write_report(report, output_dir):
    (output_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = [f"# FasterWAM components ({report['task']}, {report['num_latent_frames']} video latents)", '',
             'Same native request, checkpoint, cached text, CPU observations, noise and end-to-end wall timer '
             'as infer_video_action_random.py groups 1/2. All times in ms.', '',
             '| Component | ' + ' | '.join(f"{r['steps']} steps" for r in report['results']) + ' |',
             '|---|' + '---:|' * len(report['results'])]
    for row in ROWS:
        lines.append(f'| {row} | ' + ' | '.join(f"{r['components_ms'][row]:.3f}" for r in report['results']) + ' |')
    lines += ['', 'text_encode=0: real prompt encoding is cached outside timing, matching bench_latency.py. '
              'Components sum to the separately instrumented total; total_uninstrumented is the primary latency. '
              'other includes proprio, CPU RNG, input/output copies, schedule/mask setup and unmarked host/GPU work.', '',
              '```bash', shlex.join([sys.executable, *sys.argv]), '```', '']
    (output_dir / 'components.md').write_text('\n'.join(lines))


@torch.no_grad()
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    setup = BenchmarkSetup(args)
    report = dict(setup.metadata, timestamp=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                  initial_system=snapshot(), results=[])
    references = {steps: setup.references(steps) for steps in args.steps}
    torch.save(references, args.output_dir / 'references.pt')
    for steps in args.steps:
        request = lambda: setup.native_request(steps)
        print(f'Measuring {steps} steps, native request with cached text', flush=True)
        clean, output = benchmark(request, args.warmup, args.iterations)
        torch.testing.assert_close(output, references[steps][0], rtol=0, atol=0)
        timer = ComponentTimer()
        expected = dict(vae_encode=1, video_prepare=1, video_dit_prefill=1,
                        action_dit_denoise=steps, action_scheduler=steps)
        rows, errors = [], []
        with instrument(setup.model, timer):
            for case, reference in zip(setup.cases, references[steps]):
                timer.reset()
                actual = setup.native_request(steps, case)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                errors.append((actual - reference).abs().max().item())
            for _ in range(args.warmup):
                timer.reset()
                request()
                torch.cuda.synchronize()
            for _ in range(args.iterations):
                torch.cuda.synchronize()
                timer.reset()
                start = time.perf_counter()
                timer.origin.record()
                request()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                if timer.counts != expected:
                    raise RuntimeError(f'Unexpected calls: {timer.counts}, expected {expected}')
                rows.append(timer.values(elapsed))
        components = {key: statistics.fmean(row[key] for row in rows) for key in ROWS if key != 'total_uninstrumented'}
        components['total_uninstrumented'] = clean['summary']['mean_ms']
        report['results'].append(dict(steps=steps, components_ms=components, clean=clean,
                                      profiled=dict(component_samples_ms=rows, summary=summary([r['total'] for r in rows])),
                                      component_calls=expected, validation_max_abs_errors=errors))
        write_report(report, args.output_dir)
        print(f"{steps} steps: {clean['summary']['mean_ms']:.3f} ms", flush=True)
    print((args.output_dir / 'components.md').read_text())


if __name__ == '__main__':
    main()
