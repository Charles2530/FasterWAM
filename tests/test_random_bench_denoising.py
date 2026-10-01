"""Check that the ten-step baseline evolves action state rather than replaying it."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from infer_video_action_random import denoise_action
from fasterwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


class ActionDenoisingTest(unittest.TestCase):
    def test_ten_steps_update_state_and_restart_each_request(self):
        scheduler = WanContinuousFlowMatchScheduler(shift=5.0)
        times, deltas = scheduler.build_inference_schedule(10, torch.device("cpu"), torch.float32)
        initial = torch.ones(1, 2, 3)
        calls = []

        def predict(latents, timestep):
            calls.append((latents.clone(), timestep.item()))
            return latents  # dx/dsigma=x, so each discrete step multiplies by 1+delta.

        result = denoise_action(initial, times, deltas, predict, scheduler)
        self.assertEqual(len(calls), 10)
        self.assertEqual([t for _, t in calls], times.tolist())
        for step, (latents, _) in enumerate(calls):
            torch.testing.assert_close(latents, initial * (1 + deltas[:step]).prod())
        torch.testing.assert_close(result, initial * (1 + deltas).prod())
        torch.testing.assert_close(initial, torch.ones_like(initial))
        torch.testing.assert_close(denoise_action(initial, times, deltas, predict, scheduler), result)
        single_times, single_deltas = scheduler.build_inference_schedule(1, torch.device("cpu"), torch.float32)
        one_step = denoise_action(initial, single_times, single_deltas, predict, scheduler)
        self.assertFalse(torch.equal(result, one_step))


if __name__ == "__main__":
    unittest.main()
