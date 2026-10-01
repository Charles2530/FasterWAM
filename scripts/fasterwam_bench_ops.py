"""Benchmark-only operators ported from FastWAM latency groups 7/13/14.

Preserves BF16 intermediate rounding and FP64 rotary arithmetic. Paired RMSNorm
uses a fused Triton reduction; attention uses the default backend selection.
Supports batch-one synthetic observations with trained model weights; does not
patch model or global operators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fasterwam.models.wan22.wan_video_dit import flash_attention, rope_apply


@triton.jit
def _rms_pair(Q, K, WQ, WK, OQ, OK, NQ: tl.constexpr, D: tl.constexpr,
              QS: tl.constexpr, KS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    is_q = row < NQ
    local_row = row if is_q else row - NQ
    x_ptr = Q if is_q else K
    w_ptr = WQ if is_q else WK
    y_ptr = OQ if is_q else OK
    stride = QS if is_q else KS
    col = tl.arange(0, BLOCK)
    dtype = Q.dtype.element_ty
    x = tl.load(x_ptr + local_row * stride + col, col < D, 0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, axis=0) / D + EPS)
    normalized = (x * scale).to(dtype).to(tl.float32)
    weight = tl.load(w_ptr + col, col < D, 0).to(tl.float32)
    tl.store(y_ptr + local_row * D + col, normalized * weight, col < D)

def rms_pair(q, k, nq, nk):
    if any(x.ndim != 3 or x.shape[0] != 1 or x.stride(-1) != 1 for x in (q, k)):
        raise ValueError("Benchmark RMSNorm requires batch-one row-contiguous input")
    if q.shape[-1] != k.shape[-1] or nq.eps != nk.eps:
        raise ValueError("Paired RMSNorm requires equal widths and eps")
    oq, ok = (torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k))
    d = q.shape[-1]
    _rms_pair[(q.numel() // d + k.numel() // d,)](
        q, k, nq.weight, nk.weight, oq, ok, q.numel() // d, d,
        q.stride(1), k.stride(1), nq.eps, triton.next_power_of_2(d),
        enable_fp_fusion=False)
    return oq, ok

@triton.jit
def _rope_fp64(X, FREQ, Y, PAIRS: tl.constexpr, D: tl.constexpr,
               S: tl.constexpr, XS: tl.constexpr, HD: tl.constexpr,
               BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < PAIRS
    row, col = i // (D // 2), (i % (D // 2)) * 2
    offset = (row % S) * HD + col % HD
    real = tl.load(FREQ + offset, valid, 0).to(tl.float64)
    imag = tl.load(FREQ + offset + 1, valid, 0).to(tl.float64)
    x0 = tl.load(X + row * XS + col, valid, 0).to(tl.float64)
    x1 = tl.load(X + row * XS + col + 1, valid, 0).to(tl.float64)
    # PyTorch's CUDA cast from double to BF16/FP16 goes through float32.
    tl.store(Y + row * D + col, (x0 * real - x1 * imag).to(tl.float32), valid)
    tl.store(Y + row * D + col + 1, (x0 * imag + x1 * real).to(tl.float32), valid)

def rope(x, freqs, head_dim=128):
    if x.ndim != 3 or x.shape[0] != 1 or x.stride(-1) != 1 or not freqs.is_complex():
        raise ValueError("RoPE requires [1,S,D] row-contiguous input and complex frequencies")
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _rope_fp64[(triton.cdiv(x.numel() // 2, 256),)](
        x, torch.view_as_real(freqs), output, x.numel() // 2, x.shape[-1],
        x.shape[1], x.stride(1), head_dim, 256, enable_fp_fusion=False)
    return output

class PackedLinear(nn.Module):
    """One GEMM, with strided views into its output instead of split copies."""

    def __init__(self, linears):
        super().__init__()
        self.widths = tuple(layer.out_features for layer in linears)
        self.register_buffer("weight", torch.cat([layer.weight.detach() for layer in linears]))
        biases = [layer.bias for layer in linears]
        bias = None if all(b is None for b in biases) else torch.cat([
            layer.weight.new_zeros(layer.out_features) if layer.bias is None else layer.bias.detach()
            for layer in linears])
        self.register_buffer("bias", bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias).split(self.widths, dim=-1)

def is_unmasked(mask):
    """Only boolean all-True masks are removable; call outside graph capture."""
    return mask is None or (mask.dtype == torch.bool and bool(mask.all().item()))


class BenchmarkOps:
    VARIANTS = ("affine", "norm_rope", "unmasked", "five_ops")

    def __init__(self, mot, contexts, masks, variant):
        if variant not in self.VARIANTS:
            raise ValueError(variant)
        self.mot = mot
        self.fused_norm = variant != "affine"
        self.packed = variant == "five_ops"
        self._setup_fused_ops()
        remove_masks = variant in ("unmasked", "five_ops")
        self.masks = {id(m): None if remove_masks and is_unmasked(m) else m
                      for m in masks}
        self.contexts = {
            name: {"context": payload["context"],
                   "mask": None if remove_masks and is_unmasked(payload["mask"])
                   else payload["mask"]}
            for name, payload in contexts.items()
        }
        self.projections = {}
        if self.packed:
            for expert in mot.mixtures.values():
                for block in expert.blocks:
                    attn = block.self_attn
                    self.projections[id(attn)] = PackedLinear([attn.q, attn.k, attn.v])
                    # Sparse Action refinement blocks have no cross attention.
                    if hasattr(block, "cross_attn"):
                        attn = block.cross_attn
                        self.projections[id(attn)] = PackedLinear([attn.k, attn.v])

    def norm_pair(self, attn, q, k, freqs=None):
        if self.fused_norm:
            q, k = rms_pair(q, k, attn.norm_q, attn.norm_k)
        else:
            q, k = attn.norm_q(q), attn.norm_k(k)
        if freqs is not None:
            if self.fused_norm:
                q, k = (rope(x, freqs, attn.attn_head_dim) for x in (q, k))
            else:
                q, k = (rope_apply(x, freqs, attn.num_heads) for x in (q, k))
        return q, k

    def build(self, name, current, freqs, t_mod, layer):
        block = self.mot.mixtures[name].blocks[layer]
        x = current[name]
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = self.mot._split_modulation(
            block, t_mod[name])
        z = self._modulate(block.norm1(x), shift, scale)
        attn = block.self_attn
        q, k, v = (self.projections[id(attn)](z) if self.packed
                   else (attn.q(z), attn.k(z), attn.v(z)))
        q, k = self.norm_pair(attn, q, k, freqs[name])
        context = self.contexts[name] if name == "video" or layer in self.mot.condition_layer_set else None
        return dict(q=q, k=k, v=v, block=block, residual_x=x, gate_msa=gate,
                    shift_mlp=shift_mlp, scale_mlp=scale_mlp, gate_mlp=gate_mlp,
                    context_payload=context)

    def finish(self, io, mask, video_kv=None):
        block = io["block"]
        k, v = io["k"], io["v"]
        if video_kv is not None:
            k = torch.cat((video_kv["k"], k), dim=1)
            v = torch.cat((video_kv["v"], v), dim=1)
        mixed = flash_attention(io["q"], k, v, block.num_heads,
                                self.masks.get(id(mask), mask))
        x = self._gate(io["residual_x"], io["gate_msa"], block.self_attn.o(mixed))
        payload = io["context_payload"]
        if payload is not None:
            attn = block.cross_attn
            q = attn.q(block.norm3(x))
            k, v = (self.projections[id(attn)](payload["context"]) if self.packed
                    else (attn.k(payload["context"]), attn.v(payload["context"])))
            q, k = self.norm_pair(attn, q, k)
            mask = payload["mask"]
            if mask is not None and mask.ndim == 3:
                mask = mask.unsqueeze(1)
            x = x + attn.o(flash_attention(q, k, v, attn.num_heads, mask))
        z = self._modulate(block.norm2(x), io["shift_mlp"], io["scale_mlp"])
        return self._gate(x, io["gate_mlp"], block.ffn(z))

    def _setup_fused_ops(self):
        @triton.jit
        def affine_kernel(X, S, G, Y, N: tl.constexpr, D: tl.constexpr,
                          SROWS: tl.constexpr, GROWS: tl.constexpr,
                          SSTRIDE: tl.constexpr, GSTRIDE: tl.constexpr,
                          MODULATE: tl.constexpr, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            valid = i < N
            dtype = X.dtype.element_ty
            x = tl.load(X + i, valid, 0).to(tl.float32)
            s = tl.load(S + (i // D % SROWS) * SSTRIDE + i % D, valid, 0).to(tl.float32)
            g = tl.load(G + (i // D % GROWS) * GSTRIDE + i % D, valid, 0).to(tl.float32)
            # Preserve eager BF16/FP16 rounding after every arithmetic step.
            if MODULATE:
                factor = (1.0 + g).to(dtype).to(tl.float32)
                product = (x * factor).to(dtype).to(tl.float32)
                y = product + s
            else:
                product = (s * g).to(dtype).to(tl.float32)
                y = x + product
            tl.store(Y + i, y.to(dtype), valid)

        def supported(x, *values):
            # Modulation tensors can be strided slices with contiguous rows.
            return (x.dtype in (torch.bfloat16, torch.float16)
                    and x.ndim == 3 and x.shape[0] == 1 and x.is_contiguous()
                    and all(v.dtype == x.dtype and v.device == x.device
                            and v.ndim in (2, 3) and v.shape[-1] == x.shape[-1]
                            and v.stride(-1) == 1
                            and (v.ndim == 2 or v.shape[0] == 1)
                            and v.numel() // x.shape[-1] in (1, x.shape[1])
                            for v in values))

        def modulate(x, shift, scale):
            if not supported(x, shift, scale):
                return x * (1 + scale) + shift
            y = torch.empty_like(x)
            affine_kernel[(triton.cdiv(x.numel(), 256),)](
                x, shift, scale, y, x.numel(), x.shape[-1], shift.numel() // x.shape[-1],
                scale.numel() // x.shape[-1], shift.stride(-2), scale.stride(-2), True, 256,
                enable_fp_fusion=False)
            return y

        def gate(x, g, residual):
            if not supported(x, residual, g):
                return x + g * residual
            y = torch.empty_like(x)
            affine_kernel[(triton.cdiv(x.numel(), 256),)](
                x, residual, g, y, x.numel(), x.shape[-1], residual.numel() // x.shape[-1],
                g.numel() // x.shape[-1], residual.stride(-2), g.stride(-2), False, 256,
                enable_fp_fusion=False)
            return y

        self._modulate, self._gate = modulate, gate
