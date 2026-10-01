"""Request boundary checks without loading checkpoints or requiring a GPU."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fasterwam_benchmark import ActionRunner, BenchmarkSetup, timed_request


class BenchmarkContractTest(unittest.TestCase):
    def test_validation_tolerances_and_diagnostic_failures(self):
        class InaccurateRunner(ActionRunner):
            def __init__(self):
                self.setup = SimpleNamespace(cases=[(None, None, seed) for seed in (42, 43, 44)],
                                             video_shape=(1, 1, 3))
                self.future_seed = None
                self.error = 0.03125
                self.future_drift = 0.0

            def stage_inputs(self):
                pass

            def compute(self):
                return torch.tensor([self.error if self.future_seed is None else 0.25])

            def __call__(self):
                return self.compute() + (self.future_drift if self.future_seed is not None else 0)

        runner = InaccurateRunner()
        references = [torch.zeros(1) for _ in runner.setup.cases]
        self.assertEqual(runner.verify(references), [0.03125] * 3)
        with self.assertRaises(AssertionError):
            runner.verify(references, rtol=0.01, atol=0.01)
        failures = []
        errors = runner.verify(references, rtol=0.01, atol=0.01, validation_failures=failures)
        self.assertEqual(len(errors), 3)
        self.assertEqual([failure['check'] for failure in failures],
                         ['native_case_0', 'native_case_1', 'native_case_2', 'restored_native_case_0'])
        self.assertTrue(all(failure['rtol'] == failure['atol'] == 0.01 for failure in failures))
        self.assertIsNone(runner.future_seed)
        self.assertEqual(runner.seed, 42)
        runner.error = 0.05
        with self.assertRaises(AssertionError):
            runner.verify(references)
        runner.error = 0.03125
        runner.future_drift = 0.001
        with self.assertRaises(AssertionError):
            runner.verify(references)

    def test_native_request_uses_cpu_observations_cached_text_and_requested_steps(self):
        setup = BenchmarkSetup.__new__(BenchmarkSetup)
        setup.args = SimpleNamespace(num_video_frames=9, sigma_shift=None)
        setup.horizon = 32
        setup.context, setup.context_mask = object(), object()
        image, proprio = torch.ones(1, 3, 2, 2), torch.zeros(1, 7)
        setup.cases = [(image, proprio, 42)]
        calls = []
        def infer(**kwargs):
            calls.append(kwargs)
            return {'action': torch.ones(32, 7)}
        setup.model = SimpleNamespace(infer_action_one_pass_future_cache=infer)
        for steps in (10, 1):
            result = setup.native_request(steps)
            call = calls[-1]
            self.assertEqual(call['num_inference_steps'], steps)
            self.assertEqual(call['num_video_frames'], 9)
            self.assertIs(call['input_image'], image)
            self.assertIs(call['proprio'], proprio)
            self.assertIs(call['context'], setup.context)
            self.assertIsNone(call['prompt'])
            self.assertEqual(call['rand_device'], 'cpu')
            self.assertEqual(result.shape, (32, 7))

    def test_staging_refreshes_noise_and_observations_with_independent_generators(self):
        runner = ActionRunner.__new__(ActionRunner)
        runner.setup = SimpleNamespace(video_shape=(1, 2, 3, 2, 2), action_shape=(1, 4, 2))
        runner.image_cpu, runner.proprio_cpu = torch.ones(1, 3, 2, 2), torch.ones(1, 2)
        runner.image, runner.proprio = torch.empty_like(runner.image_cpu), torch.empty_like(runner.proprio_cpu)
        runner.video_noise = torch.empty(runner.setup.video_shape)
        runner.action_noise = torch.empty(runner.setup.action_shape)
        runner.seed, runner.future_seed = 42, None
        runner.stage_inputs()
        video, action = runner.video_noise.clone(), runner.action_noise.clone()
        expected_action = torch.randn(runner.setup.action_shape, generator=torch.Generator().manual_seed(42))
        torch.testing.assert_close(action, expected_action, rtol=0, atol=0)
        runner.video_noise.fill_(999)
        runner.action_noise.fill_(999)
        runner.stage_inputs()
        torch.testing.assert_close(runner.video_noise, video, rtol=0, atol=0)
        torch.testing.assert_close(runner.action_noise, action, rtol=0, atol=0)
        runner.future_seed = 49
        runner.image_cpu = runner.image_cpu * 0.5
        runner.proprio_cpu = runner.proprio_cpu + 0.25
        runner.stage_inputs()
        self.assertFalse(torch.equal(runner.video_noise, video))
        torch.testing.assert_close(runner.action_noise, action, rtol=0, atol=0)
        torch.testing.assert_close(runner.image, runner.image_cpu, rtol=0, atol=0)
        torch.testing.assert_close(runner.proprio, runner.proprio_cpu, rtol=0, atol=0)

    def test_wall_timer_surrounds_the_entire_request_and_completion_sync(self):
        events = []
        def clock():
            events.append('clock')
            return 1.0 if events.count('clock') == 1 else 1.25
        def request():
            events.append('request including staging and CPU output')
            return 'cpu action'
        with patch('fasterwam_benchmark.torch.cuda.synchronize', side_effect=lambda: events.append('sync')), \
             patch('fasterwam_benchmark.time.perf_counter', side_effect=clock):
            elapsed, output = timed_request(request)
        self.assertEqual(events, ['sync', 'clock', 'request including staging and CPU output', 'sync', 'clock'])
        self.assertEqual(elapsed, 250)
        self.assertEqual(output, 'cpu action')


if __name__ == '__main__':
    unittest.main()
