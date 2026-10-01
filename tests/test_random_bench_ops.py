"""Numerical checks for the random benchmark's BF16 and strided operators."""

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fasterwam_bench_ops import BenchmarkOps, PackedLinear, rms_pair, rope
from fasterwam.models.wan22.wan_video_dit import DiTBlock, RMSNorm, flash_attention, precompute_freqs_cis, rope_apply


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
class RandomBenchmarkOpsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    @torch.inference_mode()
    def test_paired_norm_and_rope_on_packed_views(self):
        for width, seq in ((3072, 98), (3072, 294), (3072, 32), (1024, 32)):
            packed = torch.randn(1, seq, width * 3, device="cuda", dtype=torch.bfloat16)
            q, k, _ = packed.chunk(3, dim=-1)
            # Cross-attention Q/K have different sequence lengths.
            for key in (k, torch.randn(1, 129, width, device="cuda", dtype=q.dtype)):
                nq, nk = [RMSNorm(width, eps=1e-6).cuda().bfloat16() for _ in range(2)]
                nq.weight.uniform_(0.5, 1.5)
                nk.weight.uniform_(0.5, 1.5)
                oq, ok = rms_pair(q, key, nq, nk)
                torch.testing.assert_close(oq, nq(q), rtol=0.008, atol=0.001)
                torch.testing.assert_close(ok, nk(key), rtol=0.008, atol=0.001)
            freqs = precompute_freqs_cis(128, seq).cuda().unsqueeze(1)
            torch.testing.assert_close(rope(q, freqs), rope_apply(q, freqs, width // 128),
                                       rtol=0, atol=0)

    @torch.inference_mode()
    def test_affine_preserves_bf16_intermediate_rounding(self):
        ops = BenchmarkOps.__new__(BenchmarkOps)
        ops._setup_fused_ops()
        x = torch.randn(1, 32, 1024, device="cuda", dtype=torch.bfloat16)
        for rows in (1, 32):
            shift, scale, gate = torch.randn(1, rows, 3, 1024, device="cuda",
                                             dtype=x.dtype).unbind(2)
            residual = torch.randn_like(x)
            torch.testing.assert_close(ops._modulate(x, shift, scale), x * (1 + scale) + shift,
                                       rtol=0, atol=0)
            torch.testing.assert_close(ops._gate(x, gate, residual), x + gate * residual,
                                       rtol=0, atol=0)

    @torch.inference_mode()
    def test_packed_projection_preserves_each_linear(self):
        layers = [torch.nn.Linear(1024, 1024, device="cuda", dtype=torch.bfloat16)
                  for _ in range(3)]
        x = torch.randn(1, 32, 1024, device="cuda", dtype=torch.bfloat16)
        for actual, layer in zip(PackedLinear(layers)(x), layers):
            torch.testing.assert_close(actual, layer(x), rtol=0.008, atol=0.001)

    @torch.inference_mode()
    def test_only_unmasked_variants_remove_boolean_true_masks(self):
        block = DiTBlock(hidden_dim=128, attn_head_dim=128, num_heads=1,
                         ffn_dim=256, eps=1e-6).cuda().bfloat16()
        mot = SimpleNamespace(mixtures={"video": SimpleNamespace(blocks=[block])})
        x = torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16)
        zero = torch.zeros(1, 1, 128, device="cuda", dtype=x.dtype)
        for variant in BenchmarkOps.VARIANTS:
            for mask, removable in ((torch.ones(4, 4, device="cuda", dtype=torch.bool), True),
                                    (torch.eye(4, device="cuda", dtype=torch.bool), False),
                                    (torch.ones(4, 4, device="cuda", dtype=x.dtype), False)):
                contexts = {"video": {"context": x, "mask": mask.unsqueeze(0)}}
                ops = BenchmarkOps(mot, contexts, [mask], variant)
                io = dict(q=x, k=x, v=x, block=block, residual_x=x, gate_msa=zero,
                          shift_mlp=zero, scale_mlp=zero, gate_mlp=zero,
                          context_payload=ops.contexts["video"])
                with patch("fasterwam_bench_ops.flash_attention", wraps=flash_attention) as attention:
                    ops.finish(io, mask)
                self.assertEqual(attention.call_count, 2)
                if removable and variant in ("unmasked", "five_ops"):
                    self.assertIsNone(attention.call_args_list[0].args[4])
                    self.assertIsNone(attention.call_args_list[1].args[4])
                else:
                    self.assertIs(attention.call_args_list[0].args[4], mask)
                    torch.testing.assert_close(attention.call_args_list[1].args[4],
                                               mask[None, None], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
