#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
import sys
import threading
import time
from pathlib import Path
from datetime import datetime

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(PACKAGE_ROOT.parent), str(PACKAGE_ROOT.parents[1] / "holosoma")]

import numpy as np
import tyro
from loguru import logger

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values.inference import AnnotatedInferenceConfig
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.motion_paths import resolve_motion_path
from holosoma_inference.policies.wbt_two_policy import WholeBodyTrackingPolicy
from holosoma_inference.run_dual_policy import AutoSwitchDualPolicy, JointTorqueSubscriber
from holosoma_inference.utils.misc import restore_terminal_settings


class ThreePolicy(AutoSwitchDualPolicy):
    roles = ("pick", "shoot", "recovery")
    pick_match_start_frame = AutoSwitchDualPolicy.recovery_min_shooting_steps
    pick_mean_abs_threshold = AutoSwitchDualPolicy.recovery_mean_abs_threshold
    pick_max_abs_threshold = AutoSwitchDualPolicy.recovery_max_abs_threshold
    pick_match_required_frames = 1

    def __init__(self, config: InferenceConfig):
        paths = config.task.model_path
        paths = paths.replace(";", ",").split(",") if isinstance(paths, str) else list(paths)
        paths = [str(path).strip() for path in paths if str(path).strip()]
        if len(paths) != 3:
            raise ValueError("Supply exactly three WBT models in order: pick,shoot,recovery.")
        if config.task.rl_rate != 50:
            raise ValueError("The pick/shoot/recovery pipeline requires 50 Hz.")
        self._validate_environment(config)
        if config.task.use_joystick:
            raise ValueError("Use --task.no-use-joystick: this pipeline uses its keyboard state machine.")
        if config.robot.robot != "g1" or config.robot.num_joints != 29:
            raise ValueError("This pipeline requires the G1-29DoF robot configuration.")
        self._state = "NO_DAMPING"
        self._auto_switch_triggered = False
        self._ramp_started_at = None
        self._ramp_start_q = None
        self._loco_policy = None
        self._pending_shooting_request = False
        self._skip_stiff_hold_prompt = True
        self._recording = False
        self._record_file = None
        self._record_path = None
        self._record_start_time = None
        self._last_record_obs = np.zeros((1, 0), dtype=np.float32)
        self._control_lock = threading.RLock()
        self._pick_match_count = 0
        self._pick_last_match_frame = None
        self._pick_end_started = None
        self._shoot_end_started = None
        self._match_errors = None
        self._transitions = []
        self._last_command = None
        self._closed = False
        WholeBodyTrackingPolicy.__init__(self, config)
        self._deployment_mode = config.task.deployment_mode
        self._last_deployment_log = 0.0
        self._last_guarded_q = None
        self._stiff_hold_q = self.motion_command_0[:, :self.num_dofs].copy()
        self._stiff_hold_kp = self.onnx_kp.copy()
        self._stiff_hold_kd = self.onnx_kd.copy()
        self._recovery_reference_q = self._policy_states[2]["motion_command_0"][:, :self.num_dofs].copy()
        self._create_loco_policy(config)
        original_prepare = self._loco_policy.prepare_obs_for_rl

        def prepare_loco(state):
            obs = original_prepare(state)
            self._last_record_obs = obs["actor_obs"].copy()
            return obs

        self._loco_policy.prepare_obs_for_rl = prepare_loco
        self._joint_torque_sub = JointTorqueSubscriber()
        self._wbt_executor.add_node(self._joint_torque_sub)
        self.interface.command_guard = self._capture_command
        logger.info("Three-policy ready: Enter -> 8 (land) -> B (ball) -> ] -> 9 (gantry off) -> s")
        if sys.stdin.isatty():
            self._init_keyboard_handler()

    def _init_input_device(self):
        self.use_joystick = False

    def _validate_environment(self, config):
        if config.task.deployment_mode != "sim" or config.task.interface != "lo":
            raise ValueError("This new entrypoint is currently validated for sim2sim on lo only.")

    def setup_policy(self, model_path):
        super().setup_policy(model_path)
        meta = self.onnx_policy_session.get_modelmeta().custom_metadata_map
        names = json.loads(meta["dof_names"])
        if names != list(self.robot_config.dof_names):
            raise ValueError(f"Joint order mismatch: {model_path}")
        experiment = json.loads(meta["experiment_config"])
        robot = experiment["robot"]
        angles = robot["init_state"]["default_joint_angles"]
        default_q = np.array([angles[name] for name in names], dtype=np.float32).reshape(1, -1)
        control = robot["control"]
        for value in (default_q, self.onnx_kp, self.onnx_kd):
            if np.asarray(value).size != self.num_dofs or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid model default/gains: {model_path}")
        if np.any(self.onnx_kp <= 0) or np.any(self.onnx_kd < 0):
            raise ValueError(f"Invalid PD gains: {model_path}")
        scale = np.full(self.num_dofs, control["action_scale"], dtype=np.float32)
        if control.get("action_scales_by_effort_limit_over_p_gain", False):
            scale *= np.asarray(robot["dof_effort_limit_list"], dtype=np.float32) / self.onnx_kp
        if scale.shape != (self.num_dofs,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"Invalid action scales: {model_path}")
        sim = experiment["simulator"]["config"]["sim"]
        training_fps, decimation = sim["fps"], sim["control_decimation"]
        if not np.all(np.isfinite([training_fps, decimation])) or training_fps <= 0 or decimation <= 0 or int(decimation) != decimation:
            raise ValueError(f"Invalid training FPS/decimation: {model_path}")
        if sim["fps"] / sim["control_decimation"] != 50:
            raise ValueError(f"Model training frequency is not 50 Hz: {model_path}")
        self._model_contract = {"default_q": default_q, "action_scale": scale}
        motion = experiment["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]
        with np.load(resolve_motion_path(motion["motion_file"]), allow_pickle=False) as data:
            frames = len(data["joint_pos"])
            fps = float(data["fps"].reshape(-1)[0])
        if frames < 1 or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid reference length/FPS: {model_path}")
        self._model_contract["last_frame"] = int((frames - 1) * 50 / fps)
        logger.info("Model contract loaded: {} default_q={} scale={}", Path(model_path).name,
                    default_q.tolist(), scale.tolist())

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state["contract"] = self._model_contract
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self._model_contract = state["contract"]
        self.policy_action_scale = self._model_contract["action_scale"].copy()

    def _get_active_policy_default_q(self):
        return self._model_contract["default_q"]

    def _activate_policy(self, index, announce=True):
        super()._activate_policy(index, announce)
        self._apply_active_policy_gains_to_robot_config()
        self.force_zero_obj_pos_b = index == 2
        self.goal_pos_w = (np.array([self.config.task.goal_pos_w], dtype=np.float32)
                           if index == 1 else np.zeros((1, 3), dtype=np.float32))

    def _enter_shooting_ready(self):
        self._activate_policy(0)
        self._handle_start_policy()
        self.motion_clip_progressing = False
        self._state = "PICK_READY"
        logger.info("Pick ready at frame 0; press s after disabling the robot gantry with 9.")

    def _start_stage(self, index):
        previous = self._state
        self._activate_policy(index)
        self._handle_start_policy()
        self._handle_start_motion_clip()
        self._state = ("PICKING", "SHOOTING", "RECOVERY")[index]
        if index == 0:
            self._pick_match_count = 0
            self._pick_last_match_frame = None
            self._pick_end_started = None
        elif index == 1:
            self._shoot_end_started = None
        self._transitions.append({"from": previous, "to": self._state, "match": self._match_errors})
        logger.info("Transition {} -> {}; parameters switched to {}", previous, self._state, self.active_model_path)

    def handle_keyboard_button(self, keycode):
        with self._control_lock:
            if self._closed:
                return
            key = str(keycode).lower()
            if key in {"enter", "return", "\n", "\r"} and self._state == "NO_DAMPING":
                self._start_ramp(self.interface.get_low_state())
            elif key == "]":
                if self._state == "STIFF":
                    self._enter_shooting_ready()
                elif self._state == "RAMP":
                    self._pending_shooting_request = True
            elif key == "s" and self._state == "PICK_READY":
                self._start_stage(0)
            elif key == "r":
                self._toggle_recording()
            elif key == "o":
                self._pending_shooting_request = False
                self.use_policy_action = False
                self.motion_clip_progressing = False
                self._stiff_hold_q = self.interface.get_low_state()[:, 7:7 + self.num_dofs].copy()
                self._stiff_hold_kp, self._stiff_hold_kd = self.onnx_kp.copy(), self.onnx_kd.copy()
                self._state = "STIFF"
            elif self._state == "LOCO":
                self._loco_policy.handle_keyboard_button(key)

    def _pick_pose_matches(self, state):
        if self.motion_timestep < self.pick_match_start_frame:
            self._pick_match_count = 0
            self._pick_last_match_frame = None
            return False
        target = self._policy_states[1]["motion_command_0"][0, :self.num_dofs]
        error = np.abs(state[0, 7:7 + self.num_dofs] - target)
        self._match_errors = {"mean": float(error.mean()), "max": float(error.max()),
                              "joint": self.dof_names[int(error.argmax())], "frame": self.motion_timestep}
        matched = error.mean() <= self.pick_mean_abs_threshold and error.max() <= self.pick_max_abs_threshold
        if not matched:
            self._pick_match_count = 0
        elif self.motion_timestep != self._pick_last_match_frame:
            contiguous = self._pick_last_match_frame is None or self.motion_timestep == self._pick_last_match_frame + 1
            self._pick_match_count = self._pick_match_count + 1 if contiguous else 1
        self._pick_last_match_frame = self.motion_timestep if matched else None
        return self._pick_match_count >= self.pick_match_required_frames

    def rl_inference(self, state):
        if self._state == "PICKING" and self._pick_pose_matches(state):
            self._start_stage(1)
        elif self._state == "SHOOTING" and self._recovery_pose_matches(state):
            self._start_stage(2)
        if self._state == "PICKING" and self.motion_timestep >= self._model_contract["last_frame"]:
            if self._pick_end_started is None:
                self._pick_end_started = time.monotonic()
            elif time.monotonic() - self._pick_end_started > 5:
                return self._hold_unmatched_pose(state, "Pick ended but shooting pose did not match")
        elif self._state == "SHOOTING" and self.motion_timestep >= self._model_contract["last_frame"]:
            if self._shoot_end_started is None:
                self._shoot_end_started = time.monotonic()
            elif time.monotonic() - self._shoot_end_started > 5:
                return self._hold_unmatched_pose(state, "Shooting ended but recovery pose did not match")
        return self._infer_active_wbt_policy(state)

    def _hold_unmatched_pose(self, state, reason):
        previous = self._state
        if previous == "SHOOTING":
            error = np.abs(state[0, 7:7 + self.num_dofs] - self._recovery_reference_q[0])
            self._match_errors = {"mean": float(error.mean()), "max": float(error.max()),
                                  "joint": self.dof_names[int(error.argmax())], "frame": self.motion_timestep}
        self.use_policy_action = False
        self.motion_clip_progressing = False
        self._pending_shooting_request = False
        self._stiff_hold_q = state[:, 7:7 + self.num_dofs].copy()
        self._stiff_hold_kp = self.onnx_kp.copy()
        self._stiff_hold_kd = self.onnx_kd.copy()
        self._state = "STIFF"
        self._transitions.append({"from": previous, "to": "STIFF", "reason": reason, "match": self._match_errors})
        logger.warning("{}; stopping tracking and holding the measured pose, without switching models. Errors={}",
                       reason, self._match_errors)
        return self._stiff_hold_q - self._get_active_policy_default_q()

    def _infer_active_wbt_policy(self, state):
        if not self.motion_clip_progressing:
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None
        obs = self.prepare_obs_for_rl(state)
        self._last_record_obs = obs["actor_obs"].copy()
        frame = min(self.motion_timestep, self._model_contract["last_frame"])
        action, self.motion_command_t, self.ref_quat_xyzw_t = self.policy(
            {"time_step": np.array([[frame]], dtype=np.float32), "obs": obs["actor_obs"]})
        self.last_policy_action = np.clip(action, -100, 100).copy()
        self.scaled_policy_action = self.last_policy_action * self.policy_action_scale
        if self.motion_clip_progressing:
            if self.use_sim_time:
                self._update_clock()
            else:
                self.motion_timestep += 1
        return self.scaled_policy_action

    def get_current_obs_buffer_dict(self, state):
        observations = super().get_current_obs_buffer_dict(state)
        observations["goal_pos_w"] = self.goal_pos_w
        return observations

    def _capture_command(self, q, dq, tau, measured, kp, kd):
        effective_kp = np.asarray(self.interface.robot_config.motor_kp if kp is None else kp) * self.interface.kp_level
        effective_kd = np.asarray(self.interface.robot_config.motor_kd if kd is None else kd) * self.interface.kd_level
        for value in (q, dq, tau, effective_kp, effective_kd):
            if np.asarray(value).shape != (self.num_dofs,) or not np.all(np.isfinite(value)):
                raise RuntimeError("Invalid command shape or non-finite control parameter.")
        self._last_command = {"q": np.asarray(q).copy(), "dq": np.asarray(dq).copy(),
                              "tau_ff": np.asarray(tau).copy(), "kp": effective_kp, "kd": effective_kd}
        return q, dq, tau, kp, kd

    def policy_action(self):
        with self._control_lock:
            state = self.interface.get_low_state()
            if self._state in {"NO_DAMPING", "RAMP", "STIFF"}:
                self._run_ramp_or_hold(state)
            elif self._state == "LOCO":
                self._loco_policy.update_phase_time()
                action = self._loco_policy.rl_inference(state)
                self._send_hold_command(state, action + self._loco_policy.default_dof_angles,
                                        self._loco_policy.onnx_kp, self._loco_policy.onnx_kd)
            else:
                action = self.rl_inference(state)
                self._send_hold_command(state, action + self._get_active_policy_default_q(),
                                        self.onnx_kp, self.onnx_kd)
            self._record_frame(state)
            if self._state == "RECOVERY" and self.motion_timestep >= self.recovery_end_frame:
                self._switch_to_loco()

    def _toggle_recording(self):
        if self._recording:
            self._stop_recording()
            logger.info("Recording stopped: {}", self._record_path)
            return
        folder = Path(__file__).resolve().parents[3] / "logs/record"
        folder.mkdir(parents=True, exist_ok=True)
        self._record_path = folder / f"three_policy_{datetime.now():%Y%m%d_%H%M%S_%f}.txt"
        self._record_file = self._record_path.open("x", encoding="utf-8")
        self._record_start_time = time.monotonic()
        try:
            self._record_file.write(json.dumps({"type": "pipeline_models", "roles": dict(zip(self.roles, self.model_paths)),
                                           "loco": self._loco_policy.model_paths[0], "rl_rate_hz": self.config.task.rl_rate,
                                           "pick_match_start_frame": self.pick_match_start_frame,
                                           "pick_mean_abs_threshold": self.pick_mean_abs_threshold,
                                           "pick_max_abs_threshold": self.pick_max_abs_threshold,
                                               "pick_match_required_frames": self.pick_match_required_frames}) + "\n")
            self._record_file.flush()
        except Exception:
            self._stop_recording()
            raise
        self._recording = True
        logger.info("Recording started: {}", self._record_path)

    def _stop_recording(self):
        self._recording = False
        handle, self._record_file = self._record_file, None
        if handle is not None:
            try:
                handle.flush()
            finally:
                handle.close()

    def close(self):
        with self._control_lock:
            self._closed = True
            self._stop_recording()

    def _record_frame(self, state):
        if not self._recording or self._last_command is None:
            return
        command = self._last_command
        velocity = state[0, 7 + self.num_dofs + 6:7 + self.num_dofs + 6 + self.num_dofs]
        torque = command["kp"] * (command["q"] - state[0, 7:7 + self.num_dofs])
        torque += command["kd"] * (command["dq"] - velocity) + command["tau_ff"]
        loco = self._state == "LOCO"
        frame = {"type": "frame", "time_s": time.monotonic() - self._record_start_time,
                 "state": self._state, "role": "loco" if loco else self.roles[self.active_policy_index],
                 "use_policy_action": self.use_policy_action, "motion_clip_progressing": self.motion_clip_progressing,
                 "motion_timestep": self.motion_timestep, "command": command, "torque_pd_estimate": torque,
                 "default_q": self._loco_policy.default_dof_angles if loco else self._get_active_policy_default_q(),
                 "action_scale": self._loco_policy.policy_action_scale if loco else self.policy_action_scale,
                 "action_raw": self._loco_policy.last_policy_action if loco else self.last_policy_action,
                 "action_scaled": self._loco_policy.scaled_policy_action if loco else self.scaled_policy_action,
                 "obs_actor": self._last_record_obs, "robot_state": state, "match": self._match_errors,
                 "transitions": self._transitions, "object_pos_w": self.object_state_sub.latest_pos_w,
                 "object_velocity_w": self.object_state_sub.latest_lin_vel_w,
                 "measured_joint_torque": self._joint_torque_sub.latest_torque,
                 "torque_received": self._joint_torque_sub.received}
        self._record_file.write(json.dumps(frame, default=self._json_value) + "\n")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pick-match-start-frame", type=int, default=ThreePolicy.pick_match_start_frame)
    parser.add_argument("--pick-mean-abs-threshold", type=float, default=ThreePolicy.pick_mean_abs_threshold)
    parser.add_argument("--pick-max-abs-threshold", type=float, default=ThreePolicy.pick_max_abs_threshold)
    parser.add_argument("--pick-match-required-frames", type=int, default=ThreePolicy.pick_match_required_frames)
    options, remaining = parser.parse_known_args()
    if "--help" in remaining:
        parser.print_help()
    if options.pick_match_start_frame < 0 or options.pick_match_required_frames < 1:
        raise ValueError("Invalid picking match frame/count.")
    if not np.all(np.isfinite([options.pick_mean_abs_threshold, options.pick_max_abs_threshold])) or not 0 < options.pick_mean_abs_threshold <= options.pick_max_abs_threshold:
        raise ValueError("Picking thresholds must satisfy 0 < mean <= max.")
    sys.argv = [sys.argv[0], *remaining]
    config = tyro.cli(AnnotatedInferenceConfig, config=TYRO_CONFIG)
    policy = ThreePolicy(config)
    for name, value in vars(options).items():
        setattr(policy, name, value)
    try:
        policy.run()
    finally:
        policy.close()
        restore_terminal_settings()


if __name__ == "__main__":
    main()
