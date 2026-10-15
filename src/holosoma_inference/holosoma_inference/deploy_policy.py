#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from loguru import logger

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values.inference import AnnotatedInferenceConfig
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.policies.wbt_two_policy import (
    ObjectStateSub,
    TorsoLinkPosSub,
    WholeBodyTrackingPolicy,
)
from holosoma_inference.sdk.interface_wrapper import InterfaceWrapper
from holosoma_inference.utils.rate import RateLimiter
from holosoma_inference.utils.misc import restore_terminal_settings


ARM_ACKNOWLEDGEMENT = "I_UNDERSTAND_THIS_COMMANDS_A_REAL_ROBOT"


@dataclass(frozen=True)
class DeploymentOptions:
    mode: Literal["read-only", "dry-run", "armed"] = "read-only"
    ramp_seconds: float = 10.0
    max_command_step_rad: float = 0.02
    max_target_error_rad: float = 0.35
    max_joint_speed_rad_s: float = 8.0
    max_state_read_seconds: float = 0.1
    sensor_timeout_seconds: float = 0.25
    joint_limit_margin_rad: float = 0.02
    max_iterations: int | None = None
    recovery_mean_abs_threshold: float = 0.22
    recovery_max_abs_threshold: float = 0.55
    recovery_min_shooting_steps: int = 20


class ReadOnlyRobotMonitor:
    def __init__(self, config: InferenceConfig, options: DeploymentOptions):
        self.interface = InterfaceWrapper(
            config.robot,
            config.task.domain_id,
            config.task.interface,
            config.task.use_joystick,
        )
        self.num_dofs = config.robot.num_joints
        self.options = options
        self.rate = RateLimiter(config.task.rl_rate)

    def run(self) -> None:
        logger.info("READ-ONLY mode: models and tracking topics are not loaded; motor commands are never sent.")
        for iteration in itertools.count():
            started = time.monotonic()
            state = self.interface.get_low_state()
            elapsed = time.monotonic() - started
            expected = 7 + self.num_dofs + 6 + self.num_dofs
            if state is None or state.shape != (1, expected) or not np.all(np.isfinite(state)):
                raise RuntimeError(f"Invalid robot state; expected finite shape (1, {expected}).")
            if elapsed > self.options.max_state_read_seconds:
                raise RuntimeError(f"Robot state read took {elapsed:.3f}s.")
            if iteration % 50 == 0:
                quat_norm = float(np.linalg.norm(state[0, 3:7]))
                max_speed = float(np.max(np.abs(state[0, 7 + self.num_dofs + 6 :])))
                logger.info(f"read-only state healthy: quaternion_norm={quat_norm:.3f}, max_joint_speed={max_speed:.3f} rad/s")
            if self.options.max_iterations is not None and iteration + 1 >= self.options.max_iterations:
                return
            self.rate.sleep()


class TimedObjectStateSub(ObjectStateSub):
    def __init__(self, topic_name: str):
        self.last_message_time = 0.0
        super().__init__(topic_name=topic_name)

    def _callback(self, msg) -> None:
        super()._callback(msg)
        if self.received:
            self.last_message_time = time.monotonic()


class TimedTorsoLinkPosSub(TorsoLinkPosSub):
    def __init__(self, topic_name: str):
        self.last_message_time = 0.0
        super().__init__(topic_name=topic_name)

    def _callback(self, msg) -> None:
        super()._callback(msg)
        if self.received and self.quat_received:
            self.last_message_time = time.monotonic()


class SafeBasketballDeployment(WholeBodyTrackingPolicy):
    def __init__(self, config: InferenceConfig, options: DeploymentOptions):
        self.options = options
        self.deployment_state = "DISARMED"
        self.fault_reason: str | None = None
        self._ramp_started_at = 0.0
        self._ramp_start_q: np.ndarray | None = None
        self._hold_q: np.ndarray | None = None
        self._last_sent_q: np.ndarray | None = None
        self._latest_robot_q: np.ndarray | None = None
        self._deployment_action_scale: np.ndarray | None = None
        self._recovery_reference_q: np.ndarray | None = None
        self._auto_recovery_triggered = False
        self._automatic_ramp_target = False
        self._skip_stiff_hold_prompt = True

        self.object_topic = _required_topic("HOLOSOMA_OBJECT_STATE_TOPIC")
        self.torso_topic = _required_topic("HOLOSOMA_TORSO_STATE_TOPIC")
        super().__init__(config)

        self.object_state_sub.destroy_node()
        self.torso_link_pos_sub.destroy_node()
        self.object_state_sub = TimedObjectStateSub(self.object_topic)
        self.torso_link_pos_sub = TimedTorsoLinkPosSub(self.torso_topic)
        executor = getattr(self, "_wbt_executor", None)
        if executor is not None:
            executor.add_node(self.object_state_sub)
            executor.add_node(self.torso_link_pos_sub)
        self._joint_lower, self._joint_upper = self._load_joint_limits()

        if len(self.model_paths) != 2:
            raise ValueError(
                "Basketball deployment requires exactly two models in this order: "
                "--task.model-path shooting.onnx,recovery.onnx"
            )
        self._validate_model_contracts()
        recovery_motion = self._policy_states[1].get("motion_command_0")
        if recovery_motion is None:
            raise ValueError("Recovery policy did not provide a frame-zero joint reference.")
        self._recovery_reference_q = np.asarray(recovery_motion[:, : self.num_dofs], dtype=np.float64).reshape(-1)

    def setup_policy(self, model_path):
        super().setup_policy(model_path)
        self._deployment_action_scale = self._read_action_scale(model_path)

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        if self._deployment_action_scale is None:
            raise RuntimeError("Action scale was not initialized while loading the policy.")
        state["deployment_action_scale"] = self._deployment_action_scale.copy()
        state["pinocchio_robot"] = self.pinocchio_robot
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        action_scale = state.get("deployment_action_scale")
        if action_scale is None:
            raise RuntimeError("Loaded policy state does not contain an action scale.")
        scale = np.asarray(action_scale, dtype=np.float32).reshape(-1)
        if scale.shape != (self.num_dofs,):
            raise RuntimeError("Loaded policy action scale has an invalid shape.")
        self._deployment_action_scale = scale.copy()
        self.policy_action_scale = self._deployment_action_scale.reshape(1, -1)
        self.pinocchio_robot = state["pinocchio_robot"]
        if hasattr(self, "_joint_lower"):
            self._joint_lower, self._joint_upper = self._load_joint_limits()

    def _read_action_scale(self, model_path: str) -> np.ndarray:
        raw = self.onnx_policy_session.get_modelmeta().custom_metadata_map.get("action_scale")
        if raw is None:
            raw = os.environ.get("HOLOSOMA_ACTION_SCALE")
        if raw is None:
            raise ValueError(
                f"{Path(model_path).name} has no action_scale metadata. "
                "Set HOLOSOMA_ACTION_SCALE to one value or 29 comma-separated values."
            )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [float(value.strip()) for value in raw.split(",") if value.strip()]
        values = np.asarray(parsed, dtype=np.float32).reshape(-1)
        if values.size == 1:
            values = np.full(self.num_dofs, values.item(), dtype=np.float32)
        if values.size != self.num_dofs or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"Action scale for {Path(model_path).name} must contain 1 or 29 positive finite values.")
        return values

    def _load_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower_pin = np.asarray(self.pinocchio_robot.robot_model.lowerPositionLimit[7:], dtype=np.float64)
        upper_pin = np.asarray(self.pinocchio_robot.robot_model.upperPositionLimit[7:], dtype=np.float64)
        if lower_pin.size != self.num_dofs or upper_pin.size != self.num_dofs:
            raise ValueError("Could not obtain 29 joint limits from ONNX robot_urdf metadata.")
        lower = np.empty(self.num_dofs, dtype=np.float64)
        upper = np.empty(self.num_dofs, dtype=np.float64)
        for pin_index, real_index in enumerate(self.pinocchio_robot.real2pinocchio_index):
            lower[real_index] = lower_pin[pin_index]
            upper[real_index] = upper_pin[pin_index]
        return lower, upper

    def _validate_model_contracts(self) -> None:
        for index, state in enumerate(self._policy_states):
            session = state["onnx_policy_session"]
            inputs = {item.name: item for item in session.get_inputs()}
            outputs = {item.name for item in session.get_outputs()}
            if set(inputs) != {"obs", "time_step"}:
                raise ValueError(f"Policy {index + 1} ONNX inputs must be obs and time_step, got {list(inputs)}.")
            if not {"actions", "joint_pos", "joint_vel", "ref_quat_xyzw"}.issubset(outputs):
                raise ValueError(f"Policy {index + 1} is missing required WBT outputs.")
            output = session.get_outputs()[next(i for i, item in enumerate(session.get_outputs()) if item.name == "actions")]
            if output.shape[-1] not in {self.num_dofs, None, "None"}:
                raise ValueError(f"Policy {index + 1} action dimension is {output.shape[-1]}, expected 29.")
        gains = [
            (state.get("onnx_kp"), state.get("onnx_kd"))
            for state in self._policy_states
        ]
        if any(kp is None or kd is None for kp, kd in gains):
            raise ValueError("Both ONNX policies must contain finite kp and kd metadata.")
        first_kp = np.asarray(gains[0][0], dtype=np.float64)
        first_kd = np.asarray(gains[0][1], dtype=np.float64)
        second_kp = np.asarray(gains[1][0], dtype=np.float64)
        second_kd = np.asarray(gains[1][1], dtype=np.float64)
        if any(
            array.shape != (self.num_dofs,) or not np.all(np.isfinite(array))
            for array in (first_kp, first_kd, second_kp, second_kd)
        ):
            raise ValueError("Both ONNX policies must contain finite 29-element kp and kd metadata.")
        if not np.array_equal(first_kp, second_kp) or not np.array_equal(first_kd, second_kd):
            raise ValueError("Shooting and recovery ONNX KP/KD metadata must be identical for guarded switching.")

    def _active_frame_zero_q(self) -> np.ndarray:
        if self.motion_command_0 is None:
            raise RuntimeError("Active WBT policy has no frame-zero joint target.")
        return np.asarray(self.motion_command_0[:, : self.num_dofs], dtype=np.float64).reshape(-1)

    def _sensor_age(self, subscriber) -> float:
        return time.monotonic() - subscriber.last_message_time

    def _require_fresh_tracking(self) -> None:
        timeout = self.options.sensor_timeout_seconds
        if not self.object_state_sub.received or self._sensor_age(self.object_state_sub) > timeout:
            raise RuntimeError(f"Ball state on {self.object_topic} is missing or stale.")
        self._require_fresh_torso()

    def _require_fresh_torso(self) -> None:
        timeout = self.options.sensor_timeout_seconds
        if (
            not self.torso_link_pos_sub.received
            or not self.torso_link_pos_sub.quat_received
            or self._sensor_age(self.torso_link_pos_sub) > timeout
        ):
            raise RuntimeError(f"Torso pose on {self.torso_topic} is missing, incomplete, or stale.")

    def get_current_obs_buffer_dict(self, robot_state_data):
        observations = super().get_current_obs_buffer_dict(robot_state_data)
        observations["dof_pos"] = (
            robot_state_data[:, 7 : 7 + self.num_dofs] - self._active_frame_zero_q().reshape(1, -1)
        )
        self._require_fresh_torso()
        if self.active_policy_index == 0:
            if not self.object_state_sub.received or self._sensor_age(self.object_state_sub) > self.options.sensor_timeout_seconds:
                raise RuntimeError(f"Ball state on {self.object_topic} is missing or stale.")
        else:
            observations["obj_pos_b"] = np.zeros((1, 3), dtype=np.float32)
            observations["goal_pos_w"] = np.zeros((1, 3), dtype=np.float32)
        return observations

    def _validate_robot_state(self, state: np.ndarray | None, read_seconds: float) -> np.ndarray:
        expected = 7 + self.num_dofs + 6 + self.num_dofs
        if state is None or state.shape != (1, expected):
            raise RuntimeError(f"Invalid robot state shape; expected (1, {expected}).")
        if read_seconds > self.options.max_state_read_seconds:
            raise RuntimeError(f"Robot state read took {read_seconds:.3f}s.")
        if not np.all(np.isfinite(state)):
            raise RuntimeError("Robot state contains NaN or Inf.")
        quat_norm = float(np.linalg.norm(state[0, 3:7]))
        if not 0.8 <= quat_norm <= 1.2:
            raise RuntimeError(f"Invalid IMU quaternion norm: {quat_norm:.3f}.")
        velocity = state[0, 7 + self.num_dofs + 6 : expected]
        if np.max(np.abs(velocity)) > self.options.max_joint_speed_rad_s:
            raise RuntimeError("Measured joint speed exceeds deployment safety limit.")
        return state[0, 7 : 7 + self.num_dofs].astype(np.float64, copy=True)

    def _bounded_target(self, requested: np.ndarray, current: np.ndarray) -> np.ndarray:
        requested = np.asarray(requested, dtype=np.float64).reshape(-1)
        if requested.size != self.num_dofs or not np.all(np.isfinite(requested)):
            raise RuntimeError("Command target has invalid shape or contains NaN/Inf.")
        margin = self.options.joint_limit_margin_rad
        if np.any(requested < self._joint_lower + margin) or np.any(requested > self._joint_upper - margin):
            raise RuntimeError("Command target violates URDF joint limits.")
        if (
            self.options.mode == "armed"
            and self.deployment_state not in {"RAMP", "HOLD"}
            and np.max(np.abs(requested - current)) > self.options.max_target_error_rad
        ):
            raise RuntimeError("Command target is too far from the measured joint position.")
        previous = current if self._last_sent_q is None else self._last_sent_q
        step = self.options.max_command_step_rad
        return previous + np.clip(requested - previous, -step, step)

    def _begin_ramp(self) -> None:
        if self._latest_robot_q is None:
            self.logger.warning("No valid robot state received yet; ramp request rejected.")
            return
        self.use_policy_action = False
        self.motion_clip_progressing = False
        self._ramp_start_q = self._latest_robot_q.copy()
        self._hold_q = self._ramp_start_q.copy()
        self._last_sent_q = self._ramp_start_q.copy()
        self._ramp_started_at = time.monotonic()
        self.deployment_state = "RAMP"
        self.logger.warning(f"Ramping to policy {self.active_policy_index + 1} frame zero over {self.options.ramp_seconds:.1f}s")

    def _switch_policy_safely(self, index: int, automatic: bool = False) -> None:
        if index == self.active_policy_index:
            return
        self.use_policy_action = False
        self.motion_clip_progressing = False
        self._activate_policy(index)
        self._automatic_ramp_target = automatic
        self.goal_pos_w = (
            np.array([self.config.task.goal_pos_w], dtype=np.float32)
            if index == 0
            else np.zeros((1, 3), dtype=np.float32)
        )
        self._begin_ramp()

    def _recovery_pose_matches(self, robot_state: np.ndarray) -> bool:
        if self._recovery_reference_q is None or self.motion_timestep < self.options.recovery_min_shooting_steps:
            return False
        current_q = robot_state[0, 7 : 7 + self.num_dofs].astype(np.float64)
        error = np.abs(current_q - self._recovery_reference_q)
        return bool(
            np.mean(error) <= self.options.recovery_mean_abs_threshold
            and np.max(error) <= self.options.recovery_max_abs_threshold
        )

    def _policy_ready(self) -> None:
        if self.deployment_state != "HOLD":
            self.logger.warning("Policy start rejected; complete the ramp with i first.")
            return
        if self.active_policy_index == 0:
            self._require_fresh_tracking()
        self._handle_start_policy()
        self.motion_clip_progressing = False
        self.deployment_state = "POLICY_READY"

    def _start_tracking(self) -> None:
        if self.deployment_state != "POLICY_READY":
            self.logger.warning("Motion start rejected; enter POLICY_READY first.")
            return
        self._handle_start_motion_clip()
        self.deployment_state = "TRACKING"

    def _maybe_start_recovery(self, robot_state: np.ndarray) -> None:
        if self.active_policy_index != 0 or self._auto_recovery_triggered:
            return
        if self.deployment_state != "TRACKING" or not self.motion_clip_progressing:
            return
        if self._recovery_pose_matches(robot_state):
            self._auto_recovery_triggered = True
            self.logger.warning("Shooting completion pose matched; switching to recovery ramp.")
            self._switch_policy_safely(1, automatic=True)

    def _stop_to_hold(self) -> None:
        self.use_policy_action = False
        self.motion_clip_progressing = False
        if self._latest_robot_q is not None:
            self._hold_q = self._latest_robot_q.copy()
        self.deployment_state = "HOLD" if self._hold_q is not None else "DISARMED"
        self._auto_recovery_triggered = False
        self.logger.warning("Policy stopped; holding the latest measured pose.")

    def handle_keyboard_button(self, keycode):
        key = str(keycode).lower()
        if key == "i":
            self._begin_ramp()
        elif key == "]":
            self._policy_ready()
        elif key == "s":
            self._start_tracking()
        elif key in {"c", "2"}:
            self._switch_policy_safely(1)
        elif key == "1":
            self._switch_policy_safely(0)
        elif key == "o":
            self._stop_to_hold()

    def handle_joystick_button(self, cur_key):
        if cur_key == "Y":
            self._begin_ramp()
        elif cur_key == "A":
            self._policy_ready()
        elif cur_key == "start":
            self._start_tracking()
        elif cur_key == "select":
            self._switch_policy_safely(1 - self.active_policy_index)
        elif cur_key == "B":
            self._stop_to_hold()
        elif cur_key == "L1+R1":
            self.deployment_state = "FAULT"
            raise KeyboardInterrupt

    def _requested_target(self, robot_state: np.ndarray) -> np.ndarray | None:
        if self.deployment_state in {"DISARMED", "FAULT"}:
            return None
        if self.deployment_state == "RAMP":
            if self._ramp_start_q is None:
                raise RuntimeError("Ramp state has no starting position.")
            alpha = min((time.monotonic() - self._ramp_started_at) / self.options.ramp_seconds, 1.0)
            target = self._ramp_start_q + alpha * (self._active_frame_zero_q() - self._ramp_start_q)
            if alpha >= 1.0:
                self._hold_q = target.copy()
                if self._automatic_ramp_target:
                    self._handle_start_policy()
                    self._handle_start_motion_clip()
                    self.deployment_state = "RECOVERY_TRACKING"
                    self._automatic_ramp_target = False
                    self.logger.warning("Recovery ramp complete; recovery motion started automatically.")
                else:
                    self.deployment_state = "HOLD"
                    self.logger.info("Ramp complete; press ]/A to enter POLICY_READY.")
            return target
        if self.deployment_state == "HOLD":
            return self._hold_q
        scaled_action = self.rl_inference(robot_state)
        if not np.all(np.isfinite(scaled_action)):
            raise RuntimeError("Policy output contains NaN or Inf.")
        return self._active_frame_zero_q() + scaled_action.reshape(-1)

    def policy_action(self):
        started = time.monotonic()
        robot_state = self.interface.get_low_state()
        read_seconds = time.monotonic() - started
        current_q = self._validate_robot_state(robot_state, read_seconds)
        self._latest_robot_q = current_q
        if robot_state is None:
            raise RuntimeError("Robot state disappeared during validation.")

        if self.options.mode == "read-only":
            return
        self._maybe_start_recovery(np.asarray(robot_state, dtype=np.float64))
        requested = self._requested_target(np.asarray(robot_state, dtype=np.float64))
        if requested is None:
            return
        bounded = self._bounded_target(requested, current_q)
        self.cmd_q[:] = bounded
        self.cmd_dq.fill(0.0)
        self.cmd_tau.fill(0.0)
        self._last_sent_q = bounded.copy()
        if self.options.mode == "armed":
            self.interface.send_low_command(self.cmd_q, self.cmd_dq, self.cmd_tau, current_q)

    def run(self):
        self.logger.info("Controls: i/Y ramp, ]/A policy ready, s/Start motion, c/2 recovery, 1 shooting, o/B hold")
        try:
            for iteration in itertools.count():
                self.latency_tracker.start_cycle()
                if self.use_joystick and self.interface.get_joystick_msg() is not None:
                    self.process_joystick_input()
                self.policy_action()
                self.latency_tracker.end_cycle()
                if iteration % 50 == 0:
                    self.logger.info(f"state={self.deployment_state} policy={self.active_policy_index + 1} mode={self.options.mode}")
                self.rate.sleep()
        except KeyboardInterrupt:
            self.deployment_state = "FAULT"
        except Exception as exc:
            self.fault_reason = str(exc)
            self.deployment_state = "FAULT"
            logger.exception(f"Safety fault: {exc}. Motor writes are now disabled.")
            raise


def _required_topic(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must name a real ROS2 tracking topic.")
    return value


def _validate_config(config: InferenceConfig, options: DeploymentOptions) -> None:
    if config.robot.robot != "g1" or config.robot.num_joints != 29 or config.robot.sdk_type != "unitree":
        raise ValueError("Only Unitree G1-29DoF is supported.")
    if config.task.interface in {None, "", "lo"}:
        raise ValueError("Use the physical G1 network interface, not lo.")
    if config.task.use_sim_time:
        raise ValueError("Simulation time is forbidden in sim2real deployment.")
    if config.task.rl_rate != 50:
        raise ValueError("Basketball deployment requires --task.rl-rate 50.")
    if options.ramp_seconds < 5.0:
        raise ValueError("Ramp duration must be at least 5 seconds.")
    positive_limits = (
        options.max_command_step_rad,
        options.max_target_error_rad,
        options.max_joint_speed_rad_s,
        options.max_state_read_seconds,
        options.sensor_timeout_seconds,
    )
    if any(value <= 0.0 for value in positive_limits) or options.joint_limit_margin_rad < 0.0:
        raise ValueError("Deployment safety limits must be positive and the joint margin cannot be negative.")
    if options.max_iterations is not None and options.max_iterations <= 0:
        raise ValueError("--deploy-max-iterations must be positive when provided.")
    if options.mode == "armed":
        if not sys.stdin.isatty():
            raise RuntimeError("Armed deployment requires an interactive terminal.")
        logger.warning("ARMED mode can command a physical robot. Use a gantry and keep the hardware emergency stop ready.")
        acknowledgement = input(f"Type {ARM_ACKNOWLEDGEMENT} to continue: ").strip()
        if acknowledgement != ARM_ACKNOWLEDGEMENT:
            raise RuntimeError("Armed deployment acknowledgement did not match.")


def _parse_deployment_options(argv: list[str]) -> tuple[DeploymentOptions, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--deploy-mode", choices=("read-only", "dry-run", "armed"), default="read-only")
    parser.add_argument("--deploy-ramp-seconds", type=float, default=10.0)
    parser.add_argument("--deploy-max-command-step-rad", type=float, default=0.02)
    parser.add_argument("--deploy-max-target-error-rad", type=float, default=0.35)
    parser.add_argument("--deploy-max-joint-speed-rad-s", type=float, default=8.0)
    parser.add_argument("--deploy-max-state-read-seconds", type=float, default=0.1)
    parser.add_argument("--deploy-sensor-timeout-seconds", type=float, default=0.25)
    parser.add_argument("--deploy-joint-limit-margin-rad", type=float, default=0.02)
    parser.add_argument("--deploy-max-iterations", type=int, default=None)
    parser.add_argument("--deploy-recovery-mean-abs-threshold", type=float, default=0.22)
    parser.add_argument("--deploy-recovery-max-abs-threshold", type=float, default=0.55)
    parser.add_argument("--deploy-recovery-min-shooting-steps", type=int, default=20)
    known, remaining = parser.parse_known_args(argv)
    options = DeploymentOptions(
        mode=known.deploy_mode,
        ramp_seconds=known.deploy_ramp_seconds,
        max_command_step_rad=known.deploy_max_command_step_rad,
        max_target_error_rad=known.deploy_max_target_error_rad,
        max_joint_speed_rad_s=known.deploy_max_joint_speed_rad_s,
        max_state_read_seconds=known.deploy_max_state_read_seconds,
        sensor_timeout_seconds=known.deploy_sensor_timeout_seconds,
        joint_limit_margin_rad=known.deploy_joint_limit_margin_rad,
        max_iterations=known.deploy_max_iterations,
        recovery_mean_abs_threshold=known.deploy_recovery_mean_abs_threshold,
        recovery_max_abs_threshold=known.deploy_recovery_max_abs_threshold,
        recovery_min_shooting_steps=known.deploy_recovery_min_shooting_steps,
    )
    return options, remaining


def main() -> None:
    options, inference_args = _parse_deployment_options(sys.argv[1:])
    if options.mode == "read-only" and not any(
        argument == "--task.model-path" or argument.startswith("--task.model-path=")
        for argument in inference_args
    ):
        inference_args.extend(["--task.model-path", "__read_only__"])
    sys.argv = [sys.argv[0], *inference_args]
    config = tyro.cli(AnnotatedInferenceConfig, config=TYRO_CONFIG)
    _validate_config(config, options)
    if options.mode == "read-only":
        ReadOnlyRobotMonitor(config, options).run()
        return
    policy = SafeBasketballDeployment(config, options)
    policy.run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        restore_terminal_settings()
