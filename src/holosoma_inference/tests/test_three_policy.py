import json
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from holosoma_inference.config.config_values.inference import g1_29dof_wbt_160
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.run_three_policy import ThreePolicy
from holosoma_inference.run_dual_policy import AutoSwitchDualPolicy
from holosoma_inference.utils.math.quat import matrix_from_quat


MODEL_DIR = Path(__file__).resolve().parents[1] / "holosoma_inference/onnx"
MODELS = [str(MODEL_DIR / name) for name in ("pick_17000.onnx", "shoot_82000.onnx", "recover_39999.onnx")]


class FakeInterface:
    backend = "binding"
    kp_level = 1.0
    kd_level = 1.0

    def __init__(self, robot):
        self.robot_config = robot
        self.state = np.zeros((1, 71))
        self.state[0, 3] = 1
        self.sent = []

    def get_low_state(self):
        return self.state.copy()

    def send_low_command(self, q, dq, tau, measured, kp_override=None, kd_override=None):
        guard = getattr(self, "command_guard", None)
        if guard:
            q, dq, tau, kp_override, kd_override = guard(q, dq, tau, measured, kp_override, kd_override)
        self.sent.append({"q": np.asarray(q).copy(), "dq": np.asarray(dq).copy(), "tau": np.asarray(tau).copy(),
                          "kp": np.asarray(self.robot_config.motor_kp if kp_override is None else kp_override) * self.kp_level,
                          "kd": np.asarray(self.robot_config.motor_kd if kd_override is None else kd_override) * self.kd_level})


class ThreePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def init_interface(policy):
            policy.interface = FakeInterface(policy.robot_config)

        config = replace(g1_29dof_wbt_160, task=replace(g1_29dof_wbt_160.task,
                         model_path=",".join(MODELS), rl_rate=50, interface="lo", use_sim_time=False))
        with patch.object(BasePolicy, "_init_communication_components", init_interface), \
                patch.object(BasePolicy, "_init_input_device", lambda p: setattr(p, "use_joystick", False)):
            cls.policy = ThreePolicy(config)
        cls.policy._spin_wbt_ros_once = lambda max_callbacks=4: None
        cls.policy.object_state_sub.received = True
        cls.policy.object_state_sub.latest_pos_w[:] = [0.2, 0.0, 0.2]
        cls.policy.torso_link_pos_sub.received = True
        cls.policy.torso_link_pos_sub.latest_pos_w[:] = [0, 0, 0.8]

    def test_contracts_and_actual_sent_gains(self):
        p = self.policy
        for slot in range(3):
            p._start_stage(slot)
            contract = p._model_contract
            metadata = p.onnx_policy_session.get_modelmeta().custom_metadata_map
            trained = json.loads(metadata["experiment_config"])["robot"]
            names = json.loads(metadata["dof_names"])
            np.testing.assert_allclose(contract["default_q"][0],
                [trained["init_state"]["default_joint_angles"][name] for name in names])
            expected_scale = trained["control"]["action_scale"] * np.asarray(trained["dof_effort_limit_list"]) / np.asarray(json.loads(metadata["kp"]))
            np.testing.assert_allclose(contract["action_scale"], expected_scale, rtol=1e-6)
            p.interface.state[0, 7:36] = contract["default_q"][0]
            p.interface.state[0, 42:71] = np.linspace(-0.1, 0.1, 29)
            p.policy_action()
            command = p.interface.sent[-1]
            np.testing.assert_allclose(command["kp"], p.onnx_kp)
            np.testing.assert_allclose(command["kd"], p.onnx_kd)
            np.testing.assert_allclose(command["q"], p._get_active_policy_default_q()[0] + p.scaled_policy_action[0])
            np.testing.assert_allclose(p.scaled_policy_action, p.last_policy_action * contract["action_scale"])
            self.assertEqual(p._last_record_obs.shape, (1, 160))
            self.assertEqual(p.force_zero_obj_pos_b, slot == 2)
            self.assertEqual(bool(np.any(p.goal_pos_w)), slot == 1)
        p._switch_to_loco()
        p.interface.state[0, 7:36] = p._loco_policy.default_dof_angles[0]
        p.policy_action()
        command = p.interface.sent[-1]
        np.testing.assert_allclose(command["kp"], p._loco_policy.onnx_kp, rtol=1e-6)
        np.testing.assert_allclose(command["kd"], p._loco_policy.onnx_kd, rtol=1e-6)
        np.testing.assert_allclose(command["q"], (p._loco_policy.default_dof_angles + p._loco_policy.scaled_policy_action)[0])
        self.assertEqual(p._last_record_obs.shape, (1, 100))

    def test_matching_boundaries_and_auto_transitions(self):
        p = self.policy
        p._start_stage(0)
        target = p._policy_states[1]["motion_command_0"][0, :29]
        p.interface.state[0, 7:36] = target
        self.assertEqual(p.pick_match_start_frame, p.recovery_min_shooting_steps)
        self.assertEqual(p.pick_mean_abs_threshold, p.recovery_mean_abs_threshold)
        self.assertEqual(p.pick_max_abs_threshold, p.recovery_max_abs_threshold)
        self.assertEqual(p.pick_match_required_frames, 1)
        p.motion_timestep = 19
        self.assertFalse(p._pick_pose_matches(p.interface.state))
        p.motion_timestep = 20
        p.interface.state[0, 7] = target[0] + 0.6
        self.assertFalse(p._pick_pose_matches(p.interface.state))
        p.interface.state[0, 7:36] = target
        p.rl_inference(p.interface.state)
        self.assertEqual((p.active_policy_index, p._state), (1, "SHOOTING"))
        p.motion_timestep = 20
        p.interface.state[0, 7:36] = p._recovery_reference_q[0]
        p.rl_inference(p.interface.state)
        self.assertEqual((p.active_policy_index, p._state), (2, "RECOVERY"))
        p.motion_timestep = 80
        p.policy_action()
        self.assertEqual(p._state, "LOCO")

    def test_distinct_gains_are_not_inherited_from_previous_slot(self):
        p = self.policy
        model_state = p._policy_states[1]
        old_kp, old_kd = model_state["onnx_kp"], model_state["onnx_kd"]
        model_state["onnx_kp"] = old_kp * 0.8
        model_state["onnx_kd"] = old_kd * 1.2
        try:
            p._start_stage(0)
            p.policy_action()
            p._start_stage(1)
            p.policy_action()
            np.testing.assert_allclose(p.interface.sent[-1]["kp"], old_kp * 0.8)
            np.testing.assert_allclose(p.interface.sent[-1]["kd"], old_kd * 1.2)
        finally:
            model_state["onnx_kp"], model_state["onnx_kd"] = old_kp, old_kd
            p._activate_policy(0)

    def test_torso_position_not_added_twice(self):
        p = self.policy
        p._activate_policy(0)
        p.interface.state[0, 7:36] = 0
        pos, quat = p._get_ref_body_pose_in_world(p.interface.state)
        np.testing.assert_allclose(pos, [[0, 0, 0.8]])
        self.assertAlmostEqual(float(np.linalg.norm(quat)), 1)

    def test_mocap_object_observation_matches_current_dual_policy(self):
        p = self.policy
        pelvis, ball = p.pelvis_pose_sub, p.basketball_pose_sub
        saved_pos = p.object_state_sub.latest_pos_w.copy()
        try:
            pelvis.received = ball.received = True
            pelvis.latest_pos_w = np.array([[1., 2., .8]], dtype=np.float32)
            pelvis.latest_quat_xyzw = np.array([[0., 0., np.sin(.35), np.cos(.35)]], dtype=np.float32)
            ball.latest_pos_w = np.array([[1.4, 2.1, .5]], dtype=np.float32)
            p.object_state_sub.latest_pos_w[:] = [99., 99., 99.]
            for slot in range(3):
                p._activate_policy(slot)
                state = p.interface.state.copy()
                state[0, 7:36] = p._get_active_policy_default_q()[0]
                actual = p.get_current_obs_buffer_dict(state)
                dual = AutoSwitchDualPolicy.get_current_obs_buffer_dict(p, state)
                np.testing.assert_allclose(actual["obj_pos_b"], dual["obj_pos_b"])
                np.testing.assert_allclose(actual["motion_ref_ori_b"], dual["motion_ref_ori_b"])
                if slot == 2:
                    np.testing.assert_allclose(actual["obj_pos_b"], 0)
                else:
                    pos, quat = p._get_ref_body_pose_in_world(state)
                    expected = matrix_from_quat(quat)[0].T @ (ball.latest_pos_w[0] - pos[0])
                    np.testing.assert_allclose(actual["obj_pos_b"][0], expected, atol=1e-6)
                self.assertEqual(bool(np.any(actual["goal_pos_w"])), slot == 1)
        finally:
            pelvis.received = ball.received = False
            p.object_state_sub.latest_pos_w = saved_pos

    def test_recorded_torque_matches_actual_command(self):
        p = self.policy
        p._start_stage(0)
        p._toggle_recording()
        try:
            sent_start = len(p.interface.sent)
            for slot in range(3):
                p._start_stage(slot)
                p.policy_action()
            p._switch_to_loco()
            p.policy_action()
            path = p._record_path
        finally:
            p._toggle_recording()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(set(rows[0]["roles"]), {"pick", "shoot", "recovery"})
        self.assertEqual([f["role"] for f in rows[1:]], ["pick", "shoot", "recovery", "loco"])
        self.assertEqual(len(rows) - 1, len(p.interface.sent) - sent_start)
        for f, sent in zip(rows[1:], p.interface.sent[sent_start:]):
            q = np.asarray(f["robot_state"])[0, 7:36]
            dq = np.asarray(f["robot_state"])[0, 42:71]
            c = f["command"]
            for recorded, transmitted in (("q", "q"), ("dq", "dq"), ("tau_ff", "tau"), ("kp", "kp"), ("kd", "kd")):
                np.testing.assert_allclose(c[recorded], sent[transmitted])
            expected = np.asarray(c["kp"]) * (np.asarray(c["q"]) - q)
            expected += np.asarray(c["kd"]) * (np.asarray(c["dq"]) - dq) + np.asarray(c["tau_ff"])
            np.testing.assert_allclose(f["torque_pd_estimate"], expected)
        print("CONTROLLED_TEST_RECORD", path)

    def test_stop_cancels_queued_pick_start(self):
        p = self.policy
        p._state = "NO_DAMPING"
        p.handle_keyboard_button("enter")
        p.handle_keyboard_button("]")
        self.assertTrue(p._pending_shooting_request)
        p.handle_keyboard_button("o")
        p.policy_action()
        self.assertEqual(p._state, "STIFF")
        self.assertFalse(p.use_policy_action)
        self.assertFalse(p._pending_shooting_request)

    def test_unmatched_terminal_pose_stops_to_measured_hold_without_crashing(self):
        p = self.policy
        for slot, timer in ((0, "_pick_end_started"), (1, "_shoot_end_started")):
            p._start_stage(slot)
            p.motion_timestep = p._model_contract["last_frame"] + 10
            p.interface.state[0, 7:36] = 0
            setattr(p, timer, 0.0)
            measured = p.interface.state[0, 7:36].copy()
            p.policy_action()
            self.assertEqual(p._state, "STIFF")
            self.assertEqual(p.active_policy_index, slot)
            self.assertFalse(p.motion_clip_progressing)
            self.assertFalse(p.use_policy_action)
            np.testing.assert_allclose(p.interface.sent[-1]["q"], measured, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
