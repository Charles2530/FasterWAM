"""Shared FasterWAM request contract, following bench_latency.py's timing rules.

The architecture remains FasterWAM's one-pass future cache (default 3 latents).
Both entry points use the same checkpoint, synthetic CPU observations, cached
real text, native eager inference, no_grad, one CPU thread and wall-clock timer.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_VALIDATION_RTOL = 0.01
DEFAULT_VALIDATION_ATOL = 0.04


def add_common_arguments(parser):
    parser.add_argument("--task", choices=("libero", "robotwin"), default="libero")
    parser.add_argument("--checkpoint", type=Path, help="Default: the task's released FasterWAM checkpoint")
    parser.add_argument("--model-path", type=Path, default=ROOT / "checkpoints/Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--num-video-frames", type=int, default=9, help="9 RGB frames correspond to 3 latent frames")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", "--iters", dest="iterations", type=int, default=100)
    parser.add_argument("--sigma-shift", type=float, default=None)


def check_common_arguments(parser, args):
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be >=0; iterations must be >=1")
    if args.num_video_frames < 1 or args.num_video_frames % 4 != 1:
        parser.error("num-video-frames must be positive and satisfy T % 4 == 1")
    if args.sigma_shift is not None and args.sigma_shift <= 0:
        parser.error("sigma-shift must be positive")
    if args.checkpoint is None:
        name = "step_021700.pt" if args.task == "libero" else "step_029355.pt"
        args.checkpoint = ROOT / "checkpoints/fasterwam_release" / args.task / name
    for path in (args.checkpoint, args.model_path):
        if not path.exists():
            parser.error(f"Missing model path: {path}")
    return args


def summarize(samples):
    ordered = sorted(samples)
    def percentile(percent):
        position = (len(ordered) - 1) * percent / 100
        lower = int(position)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)
    return dict(mean_ms=statistics.fmean(samples), p50_ms=percentile(50), p90_ms=percentile(90),
                min_ms=min(samples), max_ms=max(samples))


def timed_request(request):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = request()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000, output


def benchmark(request, warmup, iterations):
    # Original infer_video_action_random warmup: consecutive requests, then
    # one explicit completion barrier. Warmups are excluded from samples.
    for _ in range(warmup):
        request()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        elapsed, output = timed_request(request)
        samples.append(elapsed)
    return dict(summary=summarize(samples), samples_ms=samples), output


def denoise_action(initial_latents, timesteps, deltas, predict, scheduler):
    latents = initial_latents
    for timestep, delta in zip(timesteps, deltas):
        latents = scheduler.step(predict(latents, timestep.unsqueeze(0)), delta, latents)
    return latents


def tensor_hash(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


class BenchmarkSetup:
    @torch.no_grad()
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        self.args, self.device = args, torch.device("cuda:0")
        torch.manual_seed(args.seed)
        overrides = [f"model.model_id={args.model_path.resolve()}",
                     f"model.tokenizer_model_id={args.model_path.resolve()}",
                     "model.redirect_common_files=false", "model.load_text_encoder=true",
                     "model.skip_dit_load_from_pretrain=true", "model.action_dit_pretrained_path=null"]
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            cfg = compose(config_name=f"sim_{args.task}", overrides=overrides)
        print(f"Loading {args.task} FasterWAM: {args.checkpoint}", flush=True)
        self.model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=str(self.device)).eval()
        payload = self.model.load_checkpoint(str(args.checkpoint))
        checkpoint_step = payload.get("step")
        del payload
        # Like bench_latency: reseed observations after model construction/load.
        torch.manual_seed(args.seed)
        height, width = map(int, cfg.data.train.video_size)
        self.image = torch.rand(1, 3, height, width) * 2 - 1
        self.proprio = torch.randn(1, self.model.proprio_dim)
        self.context, self.context_mask = self.model.encode_prompt(args.prompt)
        self.horizon = int(cfg.data.train.num_frames) - 1
        temporal = (args.num_video_frames - 1) // self.model.vae.temporal_downsample_factor + 1
        self.video_shape = (1, self.model.vae.model.z_dim, temporal, height // 16, width // 16)
        self.action_shape = (1, self.horizon, self.model.action_expert.action_dim)
        self.cases = [(self.image, self.proprio, args.seed),
                      (self.image * 0.5, self.proprio + 0.25, args.seed),
                      (self.image, self.proprio, args.seed + 1)]
        # An additional future-only noise change is verified on manual runners.
        self.metadata = dict(
            contract="bench_latency_end_to_end_v1", architecture="FasterWAM",
            inference_path="infer_action_one_pass_future_cache", task=args.task,
            checkpoint=str(args.checkpoint.resolve()), checkpoint_step=checkpoint_step,
            checkpoint_size=args.checkpoint.stat().st_size, model_path=str(args.model_path.resolve()),
            python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            dtype="torch.bfloat16", cpu_threads=torch.get_num_threads(),
            execution_mode="no_grad", seed=args.seed, sigma_shift=args.sigma_shift,
            image_shape=list(self.image.shape), proprio_shape=list(self.proprio.shape),
            action_shape=list(self.action_shape), video_shape=list(self.video_shape),
            video_tokens=temporal * height // 32 * (width // 32),
            num_video_frames=args.num_video_frames, num_latent_frames=temporal,
            context_shape=list(self.context.shape), prompt=args.prompt,
            input_source="synthetic_cpu_observations", rand_device="cpu",
            context_source="encode_prompt_once", text_encoder_retained=True,
            input_hashes={"image": tensor_hash(self.image), "proprio": tensor_hash(self.proprio),
                          "context": tensor_hash(self.context), "context_mask": tensor_hash(self.context_mask)},
            latency="synchronized_end_to_end_wall_ms", warmup=args.warmup, iterations=args.iterations,
            warmup_mode="consecutive_requests_then_synchronize",
            includes=["input_transfers", "cpu_noise_generation", "vae_encode", "proprio_encoder", "pre_dit",
                      "video_prefill", "kv_fusion", "schedule_construction", "action_denoising",
                      "action_projection", "scheduler_updates", "cpu_action_output"],
            excluded=["text_encoding", "model_loading", "initial_observation_generation", "graph_setup", "validation", "video_decode"],
        )
        contract = {key: self.metadata[key] for key in (
            "contract", "architecture", "inference_path", "task", "checkpoint", "checkpoint_step",
            "checkpoint_size", "model_path", "torch", "dtype", "cpu_threads", "execution_mode", "seed",
            "sigma_shift", "image_shape", "action_shape", "video_shape", "input_hashes", "includes", "excluded")}
        self.metadata["comparison_key"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"Ready: {self.video_shape}, {self.metadata['video_tokens']} video tokens; "
              f"torch={torch.__version__}, CPU threads=1, real text cached", flush=True)

    def native_request(self, steps, case=None):
        image, proprio, seed = self.cases[0] if case is None else case
        return self.model.infer_action_one_pass_future_cache(
            prompt=None, input_image=image, proprio=proprio,
            context=self.context, context_mask=self.context_mask,
            action_horizon=self.horizon, num_video_frames=self.args.num_video_frames,
            num_inference_steps=steps, sigma_shift=self.args.sigma_shift,
            seed=seed, rand_device="cpu", tiled=False)["action"]

    def references(self, steps):
        values = [self.native_request(steps, case) for case in self.cases]
        for value in values:
            if value.shape != self.action_shape[1:] or not torch.isfinite(value).all():
                raise RuntimeError("Invalid native action output")
        if any(torch.equal(values[0], other) for other in values[1:]):
            raise RuntimeError("Changed-input validation is ineffective")
        return values


class ActionRunner:
    """Staged requests with the same inputs and outputs as native FasterWAM.

CPU RNG, copies, graph replay/eager compute and CPU output are all timed.
Only graph construction/packing/validation are excluded, as in bench_latency.
"""
    def __init__(self, setup, *, pipeline=False, variant=None):
        self.setup, self.model = setup, setup.model
        self.pipeline, self.variant = pipeline, variant
        self.video, self.action, self.mot = self.model.video_expert, self.model.action_expert, self.model.mot
        self.image_cpu, self.proprio_cpu, self.seed = setup.cases[0]
        self.future_seed = None
        self.image = torch.empty_like(setup.image, device=setup.device, dtype=torch.bfloat16)
        self.proprio = torch.empty_like(setup.proprio, device=setup.device, dtype=torch.bfloat16)
        self.video_noise = torch.empty(setup.video_shape, device=setup.device, dtype=torch.bfloat16)
        self.action_noise = torch.empty(setup.action_shape, device=setup.device, dtype=torch.bfloat16)
        self.video_stream, self.action_stream = torch.cuda.Stream(), torch.cuda.Stream()
        self.ready = {layer: torch.cuda.Event() for layer in self.mot.condition_layers}
        self.stage_inputs()
        self.ops = None
        self.prepare()
        self.remove_masks = variant in ("unmasked", "five_ops")
        self.video_unmasked = bool(self.video_mask.all().item())
        self.action_unmasked = bool(self.action_mask.all().item())
        if variant is not None:
            from fasterwam_bench_ops import BenchmarkOps
            self.ops = BenchmarkOps(self.mot, self.contexts, [self.video_mask, self.action_mask], variant)
        self.run_gpu = self.compute
        self.graph = None

    def stage_inputs(self):
        # Independent generators exactly match native future-cache inference.
        vgen = torch.Generator(device="cpu").manual_seed(self.seed if self.future_seed is None else self.future_seed)
        agen = torch.Generator(device="cpu").manual_seed(self.seed)
        video = torch.randn(self.setup.video_shape, generator=vgen, dtype=torch.float32)
        action = torch.randn(self.setup.action_shape, generator=agen, dtype=torch.float32)
        self.image.copy_(self.image_cpu)
        self.proprio.copy_(self.proprio_cpu)
        self.video_noise.copy_(video)
        self.action_noise.copy_(action)

    def __call__(self):
        self.stage_inputs()
        return self.run_gpu().to(device="cpu")

    def prepare(self):
        # Do not cache image latents, proprio or schedules across requests.
        first = self.model._encode_input_image_latents_tensor(self.image, tiled=False)
        latents = self.video_noise.clone()
        latents[:, :, :1] = first.clone()
        context, mask = self.model._append_proprio_to_context(
            self.setup.context, self.setup.context_mask, self.proprio)
        vtimes, _ = self.model.infer_video_scheduler.build_inference_schedule(
            1, self.setup.device, latents.dtype, shift_override=self.setup.args.sigma_shift)
        self.atimes, self.deltas = self.model.infer_action_scheduler.build_inference_schedule(
            1, self.setup.device, self.action_noise.dtype, shift_override=self.setup.args.sigma_shift)
        self.video_pre = self.video.pre_dit(
            x=latents, timestep=vtimes[0].unsqueeze(0), context=context, context_mask=mask,
            action=None, fuse_vae_embedding_in_latents=self.video.fuse_vae_embedding_in_latents)
        self.action_pre = self.action.pre_dit(
            action_tokens=self.action_noise, timestep=self.atimes[0].unsqueeze(0), context=context, context_mask=mask)
        prepared = {"video": self.video_pre, "action": self.action_pre}
        self.tokens = {name: state["tokens"] for name, state in prepared.items()}
        self.freqs = {name: state["freqs"] for name, state in prepared.items()}
        self.t_mod = {name: state["t_mod"] for name, state in prepared.items()}
        self.contexts = {name: {"context": state["context"], "mask": state["context_mask"]}
                         for name, state in prepared.items()}
        self.video_len = self.tokens["video"].shape[1]
        per_frame = self.video_pre["meta"]["tokens_per_frame"]
        self.video_mask = self.model._build_video_attention_mask(self.video_len, per_frame, self.setup.device)
        self.attention_masks = self.model._build_mot_attention_masks(
            self.video_len, self.setup.horizon, per_frame, self.setup.device)
        self.action_mask = self.attention_masks[self.mot.condition_layers[0]][self.video_len:]
        if self.ops is not None:
            for name in prepared:
                self.ops.contexts[name]["context"] = self.contexts[name]["context"]
            # The fixed mask pattern is checked before capture; constructing
            # per-request masks remains part of the measured computation.
            self.ops.masks = {
                id(self.video_mask): None if self.remove_masks and self.video_unmasked else self.video_mask,
                id(self.action_mask): None if self.remove_masks and self.action_unmasked else self.action_mask,
            }

    def build(self, name, current, layer):
        if self.ops is not None:
            return self.ops.build(name, current, self.freqs, self.t_mod, layer)
        method = self.mot._build_video_attention_io if name == "video" else self.mot._build_action_attention_io
        return method(current, self.freqs, self.t_mod, self.contexts, layer)

    def finish(self, io, mask, kv=None):
        if self.ops is not None:
            return self.ops.finish(io, mask, kv)
        k, v = io["k"], io["v"]
        if kv is not None:
            k, v = torch.cat((kv["k"], k), dim=1), torch.cat((kv["v"], v), dim=1)
        mixed = self.mot._mixed_attention_with_num_heads(
            q_cat=io["q"], k_cat=k, v_cat=v, attention_mask=mask, num_heads=io["block"].num_heads)
        return self.mot._apply_expert_post_block(
            block=io["block"], residual_x=io["residual_x"], mixed_attn_out=mixed,
            gate_msa=io["gate_msa"], shift_mlp=io["shift_mlp"], scale_mlp=io["scale_mlp"],
            gate_mlp=io["gate_mlp"], context_payload=io["context_payload"])

    def sequential(self):
        if self.ops is None:
            cache = self.mot.prefill_video_cache(
                video_tokens=self.tokens["video"], video_freqs=self.freqs["video"],
                video_t_mod=self.t_mod["video"], video_context_payload=self.contexts["video"],
                video_attention_mask=self.video_mask)
            return self.mot.forward_action_with_video_cache(
                action_tokens=self.tokens["action"], action_freqs=self.freqs["action"],
                action_t_mod=self.t_mod["action"], action_context_payload=self.contexts["action"],
                video_kv_cache=cache, attention_mask=self.attention_masks, video_seq_len=self.video_len)
        current, interval, cache = self.tokens.copy(), [], {}
        for layer in range(self.mot.num_layers):
            io = self.build("video", current, layer)
            interval.append({"k": io["k"], "v": io["v"]})
            if layer in self.mot.condition_layer_set:
                cache[layer] = self.mot._fuse_video_kv(layer, interval)
                interval = []
            current["video"] = self.finish(io, self.video_mask)
        for layer in range(self.mot.action_num_layers):
            io = self.build("action", current, layer)
            kv = cache.get(layer)
            current["action"] = self.finish(io, self.action_mask if kv is not None else None, kv)
        return current["action"]

    def asynchronous(self):
        caller = torch.cuda.current_stream()
        self.video_stream.wait_stream(caller)
        self.action_stream.wait_stream(caller)
        current, interval = self.tokens.copy(), []
        for layer in range(self.mot.num_layers):
            conditioned = layer in self.mot.condition_layer_set
            with torch.cuda.stream(self.video_stream):
                io = self.build("video", current, layer)
                interval.append({"k": io["k"], "v": io["v"]})
                if conditioned:
                    kv = self.mot._fuse_video_kv(layer, interval)
                    interval = []
                    self.ready[layer].record(self.video_stream)
                current["video"] = self.finish(io, self.video_mask)
            with torch.cuda.stream(self.action_stream):
                io = self.build("action", current, layer)
                if conditioned:
                    self.action_stream.wait_event(self.ready[layer])
                    for value in kv.values():
                        value.record_stream(self.action_stream)
                    current["action"] = self.finish(io, self.action_mask, kv)
                else:
                    current["action"] = self.finish(io, None)
        caller.wait_stream(self.video_stream)
        caller.wait_stream(self.action_stream)
        current["action"].record_stream(caller)
        return current["action"]

    def compute(self):
        self.prepare()
        hidden = self.asynchronous() if self.pipeline else self.sequential()
        prediction = self.action.post_dit(hidden, self.action_pre)
        action = self.model.infer_action_scheduler.step(prediction, self.deltas[0], self.action_noise)
        return action[0].float()

    def capture(self, warmup):
        # Wan's scale tensors are plain CPU attributes, not registered buffers.
        # Stage these immutable model constants for capture, then restore the
        # native eager path. Image latents are still recomputed inside the graph.
        original_scale = self.model.vae.scale
        self.graph_vae_scale = [value.to(device=self.setup.device, dtype=torch.bfloat16)
                                for value in original_scale]
        self.model.vae.scale = self.graph_vae_scale
        original_freqs = [expert.freqs for expert in (self.video, self.action)]
        self.graph_freqs = [freqs.to(self.setup.device) if isinstance(freqs, torch.Tensor)
                            else tuple(value.to(self.setup.device) for value in freqs)
                            for freqs in original_freqs]
        for expert, freqs in zip((self.video, self.action), self.graph_freqs):
            expert.freqs = freqs
        self.capture_stream = torch.cuda.Stream()
        self.capture_stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(self.capture_stream):
                for _ in range(warmup):
                    self.compute()
            self.capture_stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                self.static_output = self.compute()
            self.graph.replay()
            torch.cuda.synchronize()
        finally:
            self.model.vae.scale = original_scale
            for expert, freqs in zip((self.video, self.action), original_freqs):
                expert.freqs = freqs
        def replay():
            self.graph.replay()
            return self.static_output
        self.run_gpu = replay

    def verify(self, references, *, rtol=DEFAULT_VALIDATION_RTOL,
               atol=DEFAULT_VALIDATION_ATOL, validation_failures=None):
        # Default still raises on failure. Diagnostic runs retain every failed assertion
        # without changing tolerances, so all variants can be measured/reported.
        def compare(actual, expected, check, *, rtol=rtol, atol=atol):
            try:
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            except AssertionError as error:
                if validation_failures is None:
                    raise
                validation_failures.append(dict(check=check, rtol=rtol, atol=atol,
                                                message=str(error)))

        errors = []
        try:
            for index, (case, reference) in enumerate(zip(self.setup.cases, references)):
                self.image_cpu, self.proprio_cpu, self.seed = case
                output = self()
                compare(output, reference, f"native_case_{index}")
                errors.append((output - reference).abs().max().item())
            # Change future noise alone; compare graph to its eager computation.
            self.image_cpu, self.proprio_cpu, self.seed = self.setup.cases[0]
            original = self()
            self.future_seed = self.seed + 7
            self.stage_inputs()
            expected = self.compute().cpu()
            actual = self()
            compare(actual, expected, "future_noise_graph_vs_eager", rtol=0, atol=0)
            if self.setup.video_shape[2] > 1 and torch.equal(original, actual):
                raise RuntimeError("Future latents do not affect the output")
        finally:
            self.image_cpu, self.proprio_cpu, self.seed = self.setup.cases[0]
            self.future_seed = None
        restored = self()
        compare(restored, references[0], "restored_native_case_0")
        return errors
