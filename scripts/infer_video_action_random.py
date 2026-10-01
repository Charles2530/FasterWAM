"""Thirteen FasterWAM groups using bench_latency.py's end-to-end timing contract.

The filename is retained; inputs are synthetic CPU observations, but DiT/VAE/text
weights are loaded from the real model. Text is encoded once and retained.
Every request times CPU noise generation, input copies, VAE, proprio, video/KV,
action preparation/denoising/projection, schedules and CPU action output.
All modes use no_grad and one CPU thread. Eager sequential groups call the native
FasterWAM future-cache API. Default: LIBERO checkpoint, 3 latents, seed 42,
100 warmups and 100 samples. bench_component.py shares the exact native requests.
Warmup calls run consecutively, followed by one CUDA synchronization, matching
the original script. Measured requests retain synchronized end-to-end wall timing.
Native-output validation defaults to rtol=0.01, atol=0.04 for the restored BF16
fusion path. Override with --validation-rtol / --validation-atol as needed.

python scripts/infer_video_action_random.py --output-json artifacts/13groups.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import shlex
import sys

from fasterwam_benchmark import (
    ActionRunner, BenchmarkSetup, ROOT, add_common_arguments, benchmark,
    check_common_arguments, denoise_action, torch,
    DEFAULT_VALIDATION_RTOL, DEFAULT_VALIDATION_ATOL,
)

VARIANTS = {"affine": "affine/gate", "norm_rope": "+ RMSNorm/FP64 RoPE",
            "unmasked": "+ all-True mask removal", "five_ops": "+ packed QKV/KV (five-op)"}
GROUPS = [
    dict(group_id=1, steps=10, graph=False, pipeline=False, variant=None),
    dict(group_id=2, steps=1, graph=False, pipeline=False, variant=None),
    dict(group_id=3, steps=1, graph=False, pipeline=True, variant=None),
    dict(group_id=4, steps=1, graph=True, pipeline=False, variant=None),
    dict(group_id=5, steps=1, graph=True, pipeline=True, variant=None),
] + [dict(group_id=6 + 2 * index + int(pipeline), steps=1, graph=True,
          pipeline=pipeline, variant=variant)
     for index, variant in enumerate(VARIANTS) for pipeline in (False, True)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(warmup=100)
    parser.add_argument("--graph-warmup", type=int, default=3)
    parser.add_argument("--validation-rtol", type=float, default=DEFAULT_VALIDATION_RTOL,
                        help="Native-output relative tolerance (default: %(default)s)")
    parser.add_argument("--validation-atol", type=float, default=DEFAULT_VALIDATION_ATOL,
                        help="Native-output absolute tolerance (default: %(default)s)")
    parser.add_argument("--diagnose-validation", action="store_true",
                        help="Measure all groups while recording numerical validation failures; exits 1 if any fail")
    parser.add_argument("--groups", nargs="+", type=int, default=list(range(1, 14)))
    parser.add_argument("--output-json", type=Path,
                        default=ROOT / f"artifacts/fasterwam_13groups_{datetime.now():%Y%m%d_%H%M%S}.json")
    parser.add_argument("--output-markdown", type=Path)
    args = check_common_arguments(parser, parser.parse_args())
    if any(not math.isfinite(value) or value < 0 for value in (args.validation_rtol, args.validation_atol)):
        parser.error("validation tolerances must be finite and nonnegative")
    if args.graph_warmup < 1 or len(set(args.groups)) != len(args.groups) or not set(args.groups) <= set(range(1, 14)):
        parser.error("graph-warmup must be positive; groups must be distinct IDs from 1 to 13")
    if args.output_markdown is None:
        args.output_markdown = args.output_json.with_suffix(".md")
    return args


def write_report(report, args):
    baseline = next((r['summary']['mean_ms'] for r in report['results'] if r['group_id'] == 1), None)
    for result in report['results']:
        result['speedup'] = None if baseline is None else baseline / result['summary']['mean_ms']
    lines = ["# FasterWAM: 13 end-to-end latency groups", "",
             f"Checkpoint: {report['checkpoint']}; task={report['task']}; video latents={report['num_latent_frames']}; "
             f"torch={report['torch']}; CPU threads=1; no_grad; seed={report['seed']}.", "",
             f"Warmup={report['warmup']}; samples={report['iterations']}. Text encoding is cached outside timing; "
             "CPU noise/input transfers, VAE, proprio, preparation, video/KV, action denoising, scheduler "
             "and CPU output are timed with synchronized wall time.", "",
             "FasterWAM keeps its 3-latent future-cache architecture; bench_latency.py's FastWAM "
             "first-frame architecture is not substituted. Speedup uses group 1 measured in this report.", "",
             "| Group | Action Steps | Compile (incl. VAE) | GPU Count | Execution | Fusion | Mean Latency | Speedup | Validation |",
             "|---|---:|---|---:|---|---|---:|---:|---|"]
    for r in report['results']:
        execution = 'Asynchronous' if r['pipeline'] else 'Sequential'
        if r['variant']:
            execution += ', ' + VARIANTS[r['variant']]
        speed = '-' if r['speedup'] is None else f"{r['speedup']:.2f}x"
        lines.append(f"| {r['group_id']} | {r['action_steps']} | {'CUDA Graph' if r['cuda_graph'] else 'None'} | 1 | "
                     f"{execution} | {'Yes' if r['variant'] else 'No'} | {r['summary']['mean_ms']:.3f} ms | {speed} | "
                     f"{'FAIL (diagnostic timing)' if r.get('validation_failures') else 'PASS'} |")
    lines += ["", "Only boolean all-True masks are removable; nontrivial attention masks are retained.",
              "Paired RMSNorm uses a fused Triton reduction; attention uses default backend selection after mask removal.",
              f"Native-output validation: rtol={report['validation_rtol']}, atol={report['validation_atol']}; "
              "graph/eager future-noise checks remain exact.",
              "FAIL rows are diagnostic timings and do not establish numerically equivalent speedups.",
              "", "```bash", shlex.join([sys.executable, *sys.argv]), "```", ""]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + '\n')
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text('\n'.join(lines))


@torch.no_grad()
def main():
    args = parse_args()
    setup = BenchmarkSetup(args)
    report = dict(setup.metadata, timestamp=datetime.now(timezone.utc).isoformat(),
                  command=sys.argv, graph_warmup=args.graph_warmup,
                  validation_rtol=args.validation_rtol, validation_atol=args.validation_atol,
                  diagnose_validation=args.diagnose_validation, results=[])
    selected = [group for group in GROUPS if group['group_id'] in args.groups]
    references = {steps: setup.references(steps) for steps in {group['steps'] for group in selected}}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(references, args.output_json.with_suffix('.references.pt'))
    for group in selected:
        number, steps = group['group_id'], group['steps']
        print(f"Measuring group {number}: {group}", flush=True)
        runner = None
        failures = []
        def verify(phase):
            phase_failures = [] if args.diagnose_validation else None
            values = runner.verify(references[steps], rtol=args.validation_rtol,
                                   atol=args.validation_atol, validation_failures=phase_failures)
            failures.extend(dict(phase=phase, **failure) for failure in (phase_failures or []))
            return values
        if not group['graph'] and not group['pipeline']:
            # The public native call is also used by bench_component.py.
            request = lambda: setup.native_request(steps)
            errors = [0.0] * len(setup.cases)
        else:
            runner = ActionRunner(setup, pipeline=group['pipeline'], variant=group['variant'])
            eager_errors = verify('eager')
            if group['graph']:
                runner.capture(args.graph_warmup)
            errors = verify('captured' if group['graph'] else 'eager_repeat')
            request = runner
        timing, output = benchmark(request, args.warmup, args.iterations)
        try:
            torch.testing.assert_close(output, references[steps][0],
                                       rtol=args.validation_rtol, atol=args.validation_atol)
        except AssertionError as error:
            if not args.diagnose_validation:
                raise
            failures.append(dict(phase='measured_output', check='native_case_0',
                                 rtol=args.validation_rtol, atol=args.validation_atol, message=str(error)))
        result = dict(group_id=number, action_steps=steps, cuda_graph=group['graph'],
                      pipeline=group['pipeline'], variant=group['variant'], **timing,
                      validation_max_abs_errors=errors, action_shape=list(output.shape),
                      validation_passed=not failures, validation_failures=failures)
        if runner is not None:
            result.update(eager_validation_max_abs_errors=eager_errors,
                          changed_observation_and_noise_validated=True,
                          future_only_noise_validated=setup.video_shape[2] > 1)
        report['results'].append(result)
        write_report(report, args)
        print(f"Group {number}: {timing['summary']['mean_ms']:.3f} ms; "
              f"validation={'FAIL' if failures else 'PASS'}", flush=True)
        del request, runner
        gc.collect()
        torch.cuda.empty_cache()
    print(args.output_markdown.read_text())
    print(f"Saved {args.output_json}", flush=True)
    if any(result['validation_failures'] for result in report['results']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
