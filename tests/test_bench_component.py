"""Native benchmark accounting and hook restoration, without loading weights."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_component import COMPONENTS, instrument, interval_accounting, summary


class ComponentBenchmarkTest(unittest.TestCase):
    def test_interval_union_preserves_overlap_and_unmarked_time(self):
        result = interval_accounting([
            ("vae_encode", 0, 3), ("video_dit_prefill", 4, 10),
            ("action_dit_denoise", 5, 8), ("action_dit_denoise", 9, 12)], 15)
        self.assertEqual(result["overlap_correction"], -4)
        self.assertEqual(result["other"], 4)
        self.assertEqual(sum(result[k] for k in COMPONENTS) + result["overlap_correction"]
                         + result["other"], 15)

    def test_stats_keep_extreme_samples(self):
        result = summary([1, 2, 3, 4, 100])
        self.assertEqual(result["mean_ms"], 22)
        self.assertEqual(result["max_ms"], 100)
        self.assertEqual(result["min_ms"], 1)
        self.assertEqual(result["p50_ms"], 3)

    def test_restore_hooks_even_after_exception(self):
        class Model:
            def _encode_input_image_latents_tensor(self):
                return "vae"
            encode_prompt = _encode_input_image_latents_tensor
            _predict_action_noise_with_cache = _encode_input_image_latents_tensor
        model = Model()
        original = lambda: "original"
        model.video_expert = SimpleNamespace(pre_dit=original)
        model.mot = SimpleNamespace(prefill_video_cache=original)
        model.infer_action_scheduler = SimpleNamespace(step=original)
        timer = SimpleNamespace(wrap=lambda name, fn: lambda: (name, fn()))
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with instrument(model, timer):
                self.assertEqual(model.encode_prompt(), ("text_encode", "vae"))
                raise RuntimeError("intentional")
        self.assertNotIn("encode_prompt", vars(model))
        self.assertEqual(model.encode_prompt(), "vae")
        self.assertIs(model.video_expert.pre_dit, original)


if __name__ == "__main__":
    unittest.main()
