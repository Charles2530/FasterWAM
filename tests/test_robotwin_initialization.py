"""Exercise lifecycle/error handling without importing the GPU simulator."""

import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import traceback
import tempfile
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1] / "third_party/RoboTwin"


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    if class_name:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    # Execute the real method with fake simulator dependencies.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class PlanningError(RuntimeError):
    pass


class RobotInitializationTests(unittest.TestCase):
    def test_basket_placement_failure_reaches_existing_recovery(self):
        import numpy as np

        class Arm(str):
            @property
            def opposite(self):
                return Arm('right' if self == 'left' else 'left')

        for name, placement_call in (('place_can_basket', 2), ('place_object_basket', 3)):
            play = load_function(f'envs/{name}.py', 'play_once', {
                'np': np, 'PlanningError': PlanningError,
            }, name)
            for failure_call, error in ((None, None), (placement_call, PlanningError),
                                        (1, PlanningError), (placement_call + 1, PlanningError),
                                        (placement_call, RuntimeError)):
                with self.subTest(task=name, failure_call=failure_call, error=error):
                    task = Mock(plan_success=True, arm_tag=Arm('left'), info={})
                    task.get_arm_pose.return_value = [0.] * 7
                    task.object.get_pose.return_value = SimpleNamespace(p=np.zeros(3))
                    task.basket.get_functional_point.return_value = [0., 0., 0., 1., 0., 0., 0.]
                    count = 0

                    def move(*args, **kwargs):
                        nonlocal count
                        count += 1
                        # To exercise a failed recovery, first fail placement too.
                        if count == failure_call or (failure_call == placement_call + 1 and count == placement_call):
                            task.plan_success = False
                            raise error('expert planning failed')
                        return True

                    task.move.side_effect = move
                    if failure_call in (1, placement_call + 1) or error is RuntimeError:
                        with self.assertRaises(error):
                            play(task)
                    else:
                        play(task)
                        self.assertTrue(task.plan_success)
                        if failure_call == placement_call:
                            task.move_to_pose.assert_called()
                        else:
                            task.move_to_pose.assert_not_called()

    def test_failed_move_stops_before_following_actions(self):
        move = load_function("envs/_base_task.py", "move", {
            "ArmTag": str, "Action": SimpleNamespace, "PlanningError": PlanningError,
        }, "Base_Task")
        for arms in (("left",), ("right",), ("left", "right")):
            with self.subTest(arms=arms):
                task = Mock(plan_success=True)

                def fail(**kwargs):
                    task.plan_success = False
                    return None

                task.left_move_to_pose.side_effect = fail
                task.right_move_to_pose.side_effect = fail
                task.together_move_to_pose.side_effect = fail
                actions = [(arm, [
                    SimpleNamespace(arm_tag=arm, action="move", target_pose=[0] * 7, args={}),
                    SimpleNamespace(arm_tag=arm, action="gripper", target_gripper_pos=0),
                ]) for arm in arms]
                with self.assertRaises(PlanningError):
                    move(task, *actions)
                task.set_gripper.assert_not_called()
                task.take_dense_action.assert_not_called()
        with self.assertRaises(PlanningError):
            move(SimpleNamespace(plan_success=False), (None, []))

    def test_successful_move_still_executes_following_actions(self):
        move = load_function("envs/_base_task.py", "move", {
            "ArmTag": str, "Action": SimpleNamespace, "PlanningError": PlanningError,
        }, "Base_Task")
        for arms in (("left",), ("right",), ("left", "right")):
            with self.subTest(arms=arms):
                task = Mock(plan_success=True)
                actions = [(arm, [
                    SimpleNamespace(arm_tag=arm, action="move", target_pose=[0] * 7, args={}),
                    SimpleNamespace(arm_tag=arm, action="gripper", target_gripper_pos=0),
                ]) for arm in arms]
                self.assertTrue(move(task, *actions))
                self.assertEqual(task.set_gripper.call_count, len(arms))
                self.assertEqual(task.take_dense_action.call_count, 1 if len(arms) == 2 else 2)

    def test_alarmclock_unreachable_contact_rejects_seed(self):
        play = load_function("envs/click_alarmclock.py", "play_once", {
            "ArmTag": str, "PlanningError": PlanningError,
        }, "click_alarmclock")
        task = Mock(plan_success=True)
        task.alarm.get_pose.return_value = SimpleNamespace(p=[0.1, 0, 0])
        task.get_grasp_pose.return_value = None
        with self.assertRaises(PlanningError):
            play(task)
        self.assertFalse(task.plan_success)
        task.move.assert_not_called()

    def test_unreachable_grasp_rejects_seed_without_invalid_action(self):
        action = Mock()
        grasp = load_function("envs/_base_task.py", "grasp_actor", {
            "Actor": object, "ArmTag": object, "Action": action, "PlanningError": PlanningError,
        }, "Base_Task")
        for poses in ((None, None), (None, [0] * 7), ([0] * 7, None)):
            with self.subTest(poses=poses):
                task = SimpleNamespace(plan_success=True, need_plan=True,
                                       choose_grasp_pose=Mock(return_value=poses))
                with self.assertRaises(PlanningError):
                    grasp(task, object(), "right")
                self.assertFalse(task.plan_success)
        task = SimpleNamespace(plan_success=False)
        with self.assertRaises(PlanningError):
            grasp(task, object(), "left")
        action.assert_not_called()

    def test_resume_preserves_completed_phases_and_retries_invalid_results(self):
        namespace = {"Path": Path}
        for name in ("_parse_success_rate", "_phase_result_filename", "_completed_task_rates"):
            load_function("../../experiments/robotwin/run_robotwin_manager.py", name, namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in ("complete", "partial", "broken"):
                (root / task).mkdir()
            (root / "complete/_result_clean.txt").write_text("Timestamp: test\n1.0")
            (root / "complete/_result_random.txt").write_text("Timestamp: test\n0.0")
            (root / "partial/_result_clean.txt").write_text("1.0")
            (root / "broken/_result_clean.txt").write_text("Timestamp: incomplete")
            (root / "broken/_result_random.txt").write_text("nan")
            tasks = ["complete", "partial", "broken"]
            read = namespace["_completed_task_rates"]
            fresh = read(tasks, root)
            self.assertTrue(all(value is None for rates in fresh.values() for value in rates.values()))
            resumed = read(tasks, root, resume=True)
            self.assertEqual(resumed["complete"], {"clean": 1.0, "random": 0.0})
            self.assertEqual(resumed["partial"], {"clean": 1.0, "random": None})
            self.assertEqual(resumed["broken"], {"clean": None, "random": None})

    def test_failed_initialization_is_not_reused(self):
        for failure_stage in ("set_planner", "init_joints"):
            with self.subTest(stage=failure_stage):
                bad_robot = Mock()
                failure = RuntimeError("planner initialization failed")
                getattr(bad_robot, failure_stage).side_effect = failure
                good_robot = Mock()
                good_robot.left_entity.get_links.return_value = []
                good_robot.right_entity.get_links.return_value = []
                factory = Mock(side_effect=[bad_robot, good_robot])
                load = load_function("envs/_base_task.py", "load_robot", {"Robot": factory}, "Base_Task")
                task = SimpleNamespace(scene=object(), need_topp=True)
                with self.assertRaises(RuntimeError) as caught:
                    load(task)
                self.assertIs(caught.exception, failure)
                self.assertFalse(hasattr(task, "robot"))
                load(task)
                self.assertIs(task.robot, good_robot)
                good_robot.set_planner.assert_called_once_with(task.scene)
                good_robot.init_joints.assert_called_once()
                load(task)
                good_robot.reset.assert_called_once_with(task.scene, True)
                self.assertEqual(factory.call_count, 2)

    def test_unexpected_errors_stop_both_setup_phases(self):
        class UnStableError(Exception):
            pass

        evaluate = load_function("script/eval_policy.py", "eval_policy", {
            "eval_function_decorator": lambda *args: Mock(),
            "UnStableError": UnStableError,
            "PlanningError": PlanningError,
            "np": __import__("numpy"),
            "traceback": traceback,
        })
        for rollout_phase in (False, True):
            with self.subTest(rollout_phase=rollout_phase):
                task = Mock()
                task.plan_success = True
                task.check_success.return_value = True
                failure = RuntimeError("Failed to load CUDA module")
                # An unexpected retry raises BaseException, so a regression
                # cannot trap the test in the old infinite loop.
                task.setup_demo.side_effect = ([None] if rollout_phase else []) + [failure, KeyboardInterrupt()]
                args = dict(task_name="test", policy_name="test", clear_cache_freq=5, render_freq=0)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError) as caught:
                    evaluate("test", task, args, None, 0, test_num=1)
                self.assertIs(caught.exception, failure)
                self.assertEqual(task.setup_demo.call_count, 2 if rollout_phase else 1)

    def test_unstable_scene_still_advances_seed(self):
        class UnStableError(Exception):
            pass

        evaluate = load_function("script/eval_policy.py", "eval_policy", {
            "eval_function_decorator": lambda *args: Mock(),
            "UnStableError": UnStableError,
            "PlanningError": PlanningError,
            "np": __import__("numpy"),
            "traceback": traceback,
        })
        task = Mock()
        task.setup_demo.side_effect = [UnStableError(), None, KeyboardInterrupt()]
        task.play_once.side_effect = PlanningError("No reachable grasp")
        args = dict(task_name="test", policy_name="test", clear_cache_freq=5, render_freq=3)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            evaluate("test", task, args, None, 40, test_num=1)
        self.assertEqual([call.kwargs["seed"] for call in task.setup_demo.call_args_list], [40, 41, 42])
        self.assertEqual(task.close_env.call_count, 2)

    def test_expert_linalg_failure_skips_seed_without_counting_episode(self):
        import numpy as np

        class UnStableError(Exception):
            pass

        evaluate = load_function("script/eval_policy.py", "eval_policy", {
            "eval_function_decorator": lambda *args: Mock(),
            "UnStableError": UnStableError, "PlanningError": PlanningError,
            "np": np, "traceback": traceback,
        })
        task = Mock()
        task.setup_demo.side_effect = [None, KeyboardInterrupt()]
        task.play_once.side_effect = np.linalg.LinAlgError("Eigenvalues did not converge")
        args = dict(task_name="test", policy_name="test", clear_cache_freq=5, render_freq=3)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            evaluate("test", task, args, None, 40, test_num=1)
        self.assertEqual([call.kwargs["seed"] for call in task.setup_demo.call_args_list], [40, 41])
        task.close_env.assert_called_once()
        self.assertEqual(task.test_num, 0)
        self.assertEqual(task.suc, 0)
        self.assertEqual(args["render_freq"], 0)


if __name__ == "__main__":
    unittest.main()
