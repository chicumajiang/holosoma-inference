import sys
import time
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src/holosoma_inference"), str(ROOT / "src/holosoma")]

import numpy as np
from geometry_msgs.msg import PoseStamped
from holosoma_inference.config.config_values.inference import g1_29dof_wbt_160
from holosoma_inference.deploy_tri_policy import (
    Options, TriDeployment, ReadOnly, PlacementGate, RobotInterface, reference_ball_position,
)
from holosoma_inference.utils.math.quat import matrix_from_quat

MODEL_DIR = ROOT / "src/holosoma_inference/holosoma_inference/onnx"
MODELS = [str(MODEL_DIR / name) for name in ("pick_17000.onnx", "shoot_82000.onnx", "recover_39999.onnx")]


class FakeBinding:
    def __init__(self):
        self.mode_calls = []
        self.commands = []
        self.tick = 1
        self.freeze = False
        self.q = np.zeros(29)

    def read_low_state(self):
        if not self.freeze:
            self.tick += 1
        return types.SimpleNamespace(tick=self.tick, imu=types.SimpleNamespace(quat=[1, 0, 0, 0], omega=[0, 0, 0]),
                                     motor=types.SimpleNamespace(q=self.q.copy(), dq=np.zeros(29)))

    def set_control_mode(self, mode):
        self.mode_calls.append(mode)

    def create_zero_command(self):
        return types.SimpleNamespace()

    def write_low_command(self, cmd):
        self.commands.append(cmd)


def config(mode):
    return replace(g1_29dof_wbt_160, task=replace(g1_29dof_wbt_160.task,
        model_path=",".join(MODELS), rl_rate=50, interface="eth0", use_sim_time=False,
        deployment_mode=mode, deployment_configure_control_mode=False, deployment_write_enabled=False))


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.bindings = []
        module = types.ModuleType("unitree_interface")
        module.RobotType = types.SimpleNamespace(G1=1, H1=2, H1_2=3, GO2=4)
        module.MessageType = types.SimpleNamespace(HG=1, GO2=2)
        module.ControlMode = types.SimpleNamespace(PR=1, AB=0)

        def create(*args):
            binding = FakeBinding()
            self.bindings.append(binding)
            return binding

        module.create_robot = create
        self.sdk_patch = patch.dict(sys.modules, {"unitree_interface": module})
        self.sdk_patch.start()
        self.runners = []

    def tearDown(self):
        for runner in self.runners:
            runner.close()
        self.sdk_patch.stop()

    def make_runner(self, mode):
        opts = Options(mode=mode, stable_seconds=0, max_iterations=2, loco_action_scale="0.25")
        with patch("sys.stdin.isatty", return_value=False):
            runner = TriDeployment(config(mode), opts)
        self.runners.append(runner)
        runner.tracking.refresh = lambda: None
        self.bindings[-1].q = runner._policy_states[0]["contract"]["default_q"][0].copy()
        self.seed(runner)
        return runner

    def seed(self, runner, offset=None):
        state = runner.interface.get_low_state()
        pose = {"raw_position": np.array([0, 0, 800.0]), "position": np.array([0, 0, 0.8]),
                "quat_xyzw": np.array([0, 0, 0, 1.0]), "stamp": time.time(),
                "frame_id": "mocap", "received_at": time.monotonic()}
        runner.tracking.pelvis.sample = pose
        pos, quat = runner._get_ref_body_pose_in_world(state)
        target = runner.gate.target + (np.zeros(3) if offset is None else offset)
        ball_pos = pos[0] + matrix_from_quat(quat)[0] @ target
        runner.tracking.ball.sample = dict(pose, position=ball_pos, raw_position=ball_pos * 1000)

    def enter_hold(self, runner):
        runner.handle_keyboard_button("enter")
        runner._ramp_started_at = time.monotonic() - 11
        runner.policy_action()
        self.assertEqual(runner._state, "STIFF")

    def test_dry_run_never_sets_pr_or_writes_even_after_start(self):
        runner = self.make_runner("dry-run")
        binding = self.bindings[-1]
        self.enter_hold(runner)
        runner.handle_keyboard_button("]")
        self.assertEqual(runner._state, "PICK_READY")
        try:
            runner.policy_action()
        except RuntimeError as exc:
            self.assertIn("joint limits", str(exc))
        self.assertEqual(runner._last_record_obs.shape, (1, 160))
        self.assertFalse(runner.pelvis_pose_sub.received)
        self.assertFalse(runner.basketball_pose_sub.received)
        self.assertEqual(binding.mode_calls, [])
        self.assertEqual(binding.commands, [])
        runner.close()
        self.assertEqual(binding.commands, [])

    def test_armed_sets_pr_and_writes_only_after_enter(self):
        runner = self.make_runner("armed")
        binding = self.bindings[-1]
        runner.policy_action()
        self.assertEqual(binding.mode_calls, [])
        self.assertEqual(binding.commands, [])
        self.enter_hold(runner)
        self.assertEqual(binding.mode_calls, [1])
        self.assertEqual(len(binding.commands), 1)
        runner.close()
        self.assertEqual(len(binding.commands), 2)
        np.testing.assert_allclose(binding.commands[-1].kp, 0)
        np.testing.assert_allclose(binding.commands[-1].kd, runner.options.exit_damping_kd)

    def test_misplaced_ball_blocks_bracket_and_reports_delta(self):
        runner = self.make_runner("dry-run")
        self.enter_hold(runner)
        self.seed(runner, np.array([0.10, -0.06, 0]))
        runner.handle_keyboard_button("]")
        self.assertEqual(runner._state, "STIFF")
        np.testing.assert_allclose(runner.gate.snapshot["move_in_torso"], [-0.10, 0.06, 0], atol=1e-6)
        self.seed(runner)
        runner.handle_keyboard_button("]")
        self.assertEqual(runner._state, "PICK_READY")

    def test_missing_stale_or_wrong_frame_tracking_blocks_start(self):
        runner = self.make_runner("dry-run")
        self.enter_hold(runner)
        runner.tracking.ball.sample = None
        runner.handle_keyboard_button("]")
        self.assertEqual(runner._state, "STIFF")
        self.seed(runner)
        runner.tracking.ball.sample["received_at"] -= 1
        runner.handle_keyboard_button("]")
        self.assertEqual(runner._state, "STIFF")
        self.seed(runner)
        runner.tracking.ball.sample["frame_id"] = "other_world"
        runner.handle_keyboard_button("]")
        self.assertEqual(runner._state, "STIFF")
        self.assertEqual(self.bindings[-1].commands, [])

    def test_pelvis_fk_and_ball_relative_position_invariant_to_world_shift(self):
        runner = self.make_runner("dry-run")
        state = runner.interface.get_low_state()
        self.assertTrue(runner._placement_ready(state))
        original = runner.gate.snapshot["actual_obj_pos_b"].copy()
        shift = np.array([10, -8, 2.0])
        runner.tracking.pelvis.sample["position"] += shift
        runner.tracking.ball.sample["position"] += shift
        self.assertTrue(runner._placement_ready(state))
        np.testing.assert_allclose(runner.gate.snapshot["actual_obj_pos_b"], original, atol=1e-5)

    def test_tick_detects_frozen_dds_without_relying_on_joint_motion(self):
        interface = RobotInterface(config("read-only").robot, interface_str="eth0", use_joystick=False,
                                   configure_control_mode=False, write_enabled=False)
        binding = self.bindings[-1]
        interface.get_low_state()
        with self.assertRaisesRegex(RuntimeError, "tick"):
            interface.require_fresh_state(0.25)
        interface.get_low_state()
        interface.require_fresh_state(0.25)
        binding.freeze = True
        interface.last_tick_at -= 1
        interface.get_low_state()
        with self.assertRaisesRegex(RuntimeError, "tick"):
            interface.require_fresh_state(0.25)
        self.assertEqual(binding.commands, [])

    def test_read_only_does_not_load_onnx_or_touch_control(self):
        with patch("onnxruntime.InferenceSession", side_effect=AssertionError("No model allowed")):
            monitor = ReadOnly(config("read-only"), Options(stable_seconds=0))
        self.runners.append(monitor)
        monitor.tracking.refresh = lambda: None
        self.bindings[-1].q = monitor.startup_q.copy()
        pose = {"position": np.array([0, 0, .8]), "quat_xyzw": np.array([0, 0, 0, 1.0]),
                "raw_position": np.array([0, 0, 800]), "received_at": time.monotonic(), "stamp": time.time(),
                "frame_id": "mocap"}
        monitor.tracking.pelvis.sample = pose
        state = monitor.interface.get_low_state()
        pos, quat = monitor.tracking.torso_pose(state, monitor.robot)
        monitor.tracking.ball.sample = dict(pose, position=pos[0] + matrix_from_quat(quat)[0] @ monitor.gate.target)
        snapshot = monitor.step()
        self.assertTrue(snapshot["placement"]["ok"])
        self.assertEqual(self.bindings[-1].commands, [])
        self.assertEqual(self.bindings[-1].mode_calls, [])

    def test_invalid_pose_does_not_keep_old_sample_valid(self):
        runner = self.make_runner("dry-run")
        msg = PoseStamped()
        msg.pose.position.x = float("nan")
        msg.pose.orientation.w = 1.0
        runner.tracking.ball._callback(msg)
        self.assertIsNone(runner.tracking.ball.sample)

    def test_stop_before_enter_cannot_arm_or_bypass_startup(self):
        runner = self.make_runner("armed")
        binding = self.bindings[-1]
        runner.handle_keyboard_button("o")
        runner.handle_keyboard_button("]")
        runner.policy_action()
        self.assertEqual(runner._state, "NO_DAMPING")
        self.assertEqual(binding.mode_calls, [])
        self.assertEqual(binding.commands, [])
        runner.handle_keyboard_button("enter")
        runner.handle_keyboard_button("o")
        runner.handle_keyboard_button("]")
        self.assertFalse(runner._startup_complete)
        self.assertEqual(runner._state, "STIFF")

    def test_closed_channel_cannot_be_reopened(self):
        runner = self.make_runner("armed")
        runner.close()
        runner.interface.write_enabled = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runner.interface.send_low_command(np.zeros(29), np.zeros(29), np.zeros(29))
        runner.policy_action()
        self.assertEqual(self.bindings[-1].commands, [])

    def test_private_native_write_is_blocked_in_non_armed_modes(self):
        runner = self.make_runner("dry-run")
        runner.interface.write_enabled = True
        with self.assertRaisesRegex(RuntimeError, "rejects"):
            runner.interface._send_binding_command(np.zeros(29), np.zeros(29), np.zeros(29))
        self.assertEqual(self.bindings[-1].commands, [])
        runner.interface.write_enabled = False

    def test_stability_dwell_resets_on_checking_gap(self):
        gate = PlacementGate(Options(mode="dry-run", stable_seconds=0.5), np.zeros(3))
        with patch("holosoma_inference.deploy_tri_policy.time.monotonic") as clock:
            for t in [1.0, 1.1, 1.2, 1.3, 1.4]:
                clock.return_value = t
                self.assertFalse(gate.check(np.zeros(3), True))
            clock.return_value = 1.6
            self.assertTrue(gate.check(np.zeros(3), True))
            clock.return_value = 2.0
            self.assertFalse(gate.check(np.zeros(3), True))

    def test_old_source_stamp_is_rejected_even_if_receive_time_is_new(self):
        runner = self.make_runner("dry-run")
        runner.tracking.pelvis.sample["stamp"] -= 10
        runner.tracking.ball.sample["stamp"] -= 10
        with self.assertRaisesRegex(RuntimeError, "源时间戳"):
            runner.tracking.require(ball=True)

    def test_replayed_source_stamp_does_not_refresh_sample(self):
        runner = self.make_runner("dry-run")
        node = runner.tracking.pelvis
        node.sample = None
        msg = PoseStamped()
        msg.header.frame_id = "mocap"
        msg.header.stamp.sec = int(time.time())
        msg.pose.orientation.w = 1.0
        node._callback(msg)
        previous = node.sample["received_at"]
        node._callback(msg)
        self.assertEqual(node.sample["received_at"], previous)
        self.assertIsNotNone(node.last_invalid)

    def test_loco_has_its_own_hardware_contract(self):
        runner = self.make_runner("dry-run")
        self.assertIsNot(runner._loco_contract["robot"], runner._policy_states[2]["pinocchio_robot"])
        np.testing.assert_allclose(runner._loco_policy.policy_action_scale, 0.25)
        self.assertEqual(runner._loco_contract["effort_limits"].shape, (29,))

    def test_missing_pelvis_before_enter_rejects_without_exiting(self):
        runner = self.make_runner("armed")
        runner.tracking.pelvis.sample = None
        runner._keyboard_press("enter")
        runner.policy_action()
        self.assertFalse(runner._closed)
        self.assertFalse(runner._arm_requested)
        self.assertEqual(self.bindings[-1].mode_calls, [])
        self.assertEqual(self.bindings[-1].commands, [])

    def test_repeated_loco_hold_keeps_loco_contract(self):
        runner = self.make_runner("dry-run")
        runner._startup_complete = True
        runner._arm_requested = True
        runner._state = "LOCO"
        runner._loco_policy.onnx_kp = runner._loco_policy.onnx_kp * 0.8
        runner.handle_keyboard_button("o")
        runner.handle_keyboard_button("o")
        self.assertTrue(runner._holding_loco)
        np.testing.assert_allclose(runner._stiff_hold_kp, runner._loco_policy.onnx_kp)

    def test_new_pick_ready_clears_loco_hold_contract(self):
        runner = self.make_runner("dry-run")
        runner._startup_complete = True
        runner._arm_requested = True
        runner._state = "LOCO"
        runner.handle_keyboard_button("o")
        self.assertTrue(runner._holding_loco)
        self.seed(runner)
        runner.handle_keyboard_button("]")
        self.assertEqual(runner._state, "PICK_READY")
        self.assertFalse(runner._holding_loco)

    def test_exit_with_frozen_tick_does_not_use_stale_damping_velocity(self):
        runner = self.make_runner("armed")
        self.enter_hold(runner)
        binding = self.bindings[-1]
        binding.freeze = True
        runner.interface.last_tick_at -= 1
        runner.close()
        np.testing.assert_allclose(binding.commands[-1].kp, 0)
        np.testing.assert_allclose(binding.commands[-1].kd, 0)

    def test_escape_closes_polling_keyboard_without_a_listener_thread(self):
        runner = self.make_runner("dry-run")
        runner._keyboard_fd = 123
        with patch("holosoma_inference.deploy_tri_policy.select.select", return_value=([123], [], [])), \
                patch("holosoma_inference.deploy_tri_policy.os.read", return_value=b"\x1b"):
            runner._poll_keyboard()
        self.assertTrue(runner._closed)
        self.assertEqual(self.bindings[-1].commands, [])
        self.assertFalse(hasattr(runner, "_keyboard_thread"))

    def test_cleanup_failure_does_not_skip_write_lock_or_other_cleanup(self):
        runner = self.make_runner("dry-run")
        with patch.object(runner, "_stop_recording", side_effect=OSError("injected file cleanup failure")):
            runner.close()
        self.assertTrue(runner._closed)
        self.assertTrue(runner.interface.command_closed)
        self.assertTrue(runner._tracking_closed)
        self.assertIsNone(runner._wbt_executor)
        runner.close()

    def test_failed_file_close_retains_handle_for_retry(self):
        runner = self.make_runner("dry-run")

        class FlakyFile:
            closed = False
            attempts = 0

            def flush(self):
                pass

            def close(self):
                self.attempts += 1
                if self.attempts == 1:
                    raise OSError("injected first close failure")
                self.closed = True

        handle = FlakyFile()
        runner._record_file = handle
        runner._recording = True
        runner.close()
        self.assertIs(runner._record_file, handle)
        self.assertFalse(handle.closed)
        self.assertTrue(runner.interface.command_closed)
        runner.close()
        self.assertTrue(handle.closed)
        self.assertIsNone(runner._record_file)

    def test_keyboard_failure_latches_fault_without_writes(self):
        runner = self.make_runner("dry-run")
        with patch.object(runner, "handle_keyboard_button", side_effect=RuntimeError("injected unexpected callback fault")):
            runner._keyboard_press("enter")
        self.assertTrue(runner._closed)
        self.assertIsNotNone(runner._fault)
        self.assertEqual(self.bindings[-1].commands, [])


if __name__ == "__main__":
    unittest.main()
