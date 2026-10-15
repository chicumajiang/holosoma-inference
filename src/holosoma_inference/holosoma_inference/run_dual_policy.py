#!/usr/bin/env python3
"""Run a two-stage WBT policy with automatic shooting-to-recovery switching.

The first model is the shooting policy and the second model is the recovery
policy. Both models are loaded by ``wbt_two_policy.WholeBodyTrackingPolicy``.
While the shooting motion is running, the current robot joint position is
compared with the recovery model's frame-zero ``joint_pos`` output. Once the
configured pose thresholds are met, recovery is activated automatically.

Example:
    python run_dual_policy.py inference:g1-29dof-wbt-160 \
        --task.model-path shooting.onnx,recovery.onnx \
        --task.no-use-joystick --task.use-sim-time --task.rl-rate 50 \
        --task.interface lo
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import traceback
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
import rclpy
from loguru import logger
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float32MultiArray
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values.inference import AnnotatedInferenceConfig
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.config.config_values import observation as observation_configs
from holosoma_inference.config.config_values import robot_init as robot_configs
from holosoma_inference.config.config_values import task as task_configs
from holosoma_inference.policies.locomotion import LocomotionPolicy
from holosoma_inference.policies.wbt_two_policy import (
    PELVIS_MOCAP_TOPIC,
    MocapPoseSub,
    WholeBodyTrackingPolicy,
)
from holosoma_inference.sdk.interface_wrapper import InterfaceWrapper
from holosoma_inference.utils.rate import RateLimiter
from holosoma_inference.utils.misc import restore_terminal_settings


class JointTorqueSubscriber(Node):
    def __init__(self):
        super().__init__("dual_policy_joint_torque_subscriber")
        self.latest_torque = np.zeros(29, dtype=np.float32)
        self.received = False
        self.create_subscription(Float32MultiArray, "/joint_torque", self._callback, 10)

    def _callback(self, msg: Float32MultiArray) -> None:
        self.latest_torque = np.asarray(msg.data, dtype=np.float32).copy()
        self.received = True


class ReadOnlyRobotMonitor:
    def __init__(self, config: InferenceConfig):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.config = config
        self.interface = InterfaceWrapper(
            config.robot,
            config.task.domain_id,
            config.task.interface,
            config.task.use_joystick,
            configure_control_mode=False,
            write_enabled=False,
        )
        self.ball = MocapPoseSub("/basketball/pose", "dual_policy_readonly_ball", 0.001)
        self.pelvis = MocapPoseSub(PELVIS_MOCAP_TOPIC, "dual_policy_readonly_pelvis", 0.001)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.ball)
        self.executor.add_node(self.pelvis)
        self.rate = RateLimiter(config.task.rl_rate)
        self._validate_joint_mapping()

    def _validate_joint_mapping(self) -> None:
        expected = tuple(range(self.config.robot.num_joints))
        if tuple(self.config.robot.joint2motor) != expected or tuple(self.config.robot.motor2joint) != expected:
            raise RuntimeError("READ-ONLY requires an explicit verified joint/motor mapping for Unitree state order.")
        logger.info(
            "Unitree motor-state order: "
            + ", ".join(f"{index}:{name}" for index, name in enumerate(self.config.robot.dof_names))
        )

    def run(self) -> None:
        expected = 7 + self.config.robot.num_joints + 6 + self.config.robot.num_joints
        logger.info("READ-ONLY: no ONNX policy, PR mode, or low-level command writes will be used.")
        for iteration in itertools.count():
            state = self.interface.get_low_state()
            for _ in range(4):
                self.executor.spin_once(timeout_sec=0.0)
            if state is None or state.shape != (1, expected) or not np.all(np.isfinite(state)):
                raise RuntimeError(f"Invalid robot state; expected finite shape (1, {expected}).")
            if not self.ball.received or not self.pelvis.received:
                raise RuntimeError("READ-ONLY requires both /basketball/pose and /pelvis/pose.")
            timeout = float(getattr(self.config.task, "mocap_timeout_seconds", 0.25))
            now = time.monotonic()
            if now - self.ball.last_message_time > timeout:
                raise RuntimeError("READ-ONLY basketball mocap is stale.")
            if now - self.pelvis.last_message_time > timeout:
                raise RuntimeError("READ-ONLY pelvis mocap is stale.")
            if iteration % 50 == 0:
                logger.info(
                    "read-only healthy: quaternion_norm={:.3f}, max_joint_speed={:.3f}, ball_mocap_age={:.3f}s, pelvis_mocap_age={:.3f}s".format(
                        np.linalg.norm(state[0, 3:7]),
                        np.max(np.abs(state[0, 7 + self.config.robot.num_joints + 6 :])),
                        time.monotonic() - self.ball.last_message_time,
                        time.monotonic() - self.pelvis.last_message_time,
                    )
                )
            self.rate.sleep()


class AutoSwitchDualPolicy(WholeBodyTrackingPolicy):
    """WBT two-policy runner with automatic recovery activation."""

    recovery_mean_abs_threshold: float = 0.22
    recovery_max_abs_threshold: float = 0.55
    recovery_min_shooting_steps: int = 20
    recovery_end_frame: int = 80
    ramp_duration_s: float = 2.0
    ramp_kp: float = 80.0
    ramp_kd: float = 1.0

    def __init__(self, config: InferenceConfig):
        raw_paths = config.task.model_path
        if isinstance(raw_paths, str):
            paths = [item.strip() for item in raw_paths.replace(";", ",").split(",") if item.strip()]
        else:
            paths = list(raw_paths)

        if len(paths) != 2:
            raise ValueError(
                "--task.model-path must contain exactly two paths: shooting_policy.onnx,recovery_policy.onnx"
            )

        actor_obs = config.observation.obs_dict.get("actor_obs", [])
        if "goal_pos_w" not in actor_obs:
            raise ValueError(
                "The selected inference preset must include goal_pos_w in actor_obs. "
                "Use inference:g1-29dof-wbt-160 for these 160-dim models."
            )

        self._auto_switch_triggered: bool = False
        self._state: Literal["NO_DAMPING", "RAMP", "STIFF", "LOCO", "SHOOT_READY", "SHOOTING", "RECOVERY"] = "NO_DAMPING"
        self._ramp_started_at: float | None = None
        self._ramp_start_q: np.ndarray | None = None
        self._loco_policy: LocomotionPolicy | None = None
        self._pending_shooting_request: bool = False
        self._skip_stiff_hold_prompt = True
        self._recording = False
        self._record_file = None
        self._record_path: Path | None = None
        self._record_start_time: float | None = None
        self._last_record_obs = np.zeros((1, 0), dtype=np.float32)
        super().__init__(config)
        self._deployment_mode = getattr(config.task, "deployment_mode", "sim")
        self._last_deployment_log = 0.0
        self._last_guarded_q: np.ndarray | None = None
        self.interface.command_guard = self._guard_command
        expected = tuple(range(self.num_dofs))
        if tuple(config.robot.joint2motor) != expected or tuple(config.robot.motor2joint) != expected:
            raise RuntimeError("Unitree joint/motor mapping is not the verified identity mapping.")

        recovery_motion_command_0 = self._policy_states[1].get("motion_command_0")
        if recovery_motion_command_0 is None:
            raise ValueError("Recovery policy did not provide a reference frame-0 joint position.")
        self._recovery_reference_q: np.ndarray = np.asarray(
            recovery_motion_command_0[:, : self.num_dofs], dtype=np.float32
        ).copy()
        shooting_motion_command_0 = self._policy_states[0].get("motion_command_0")
        if shooting_motion_command_0 is None:
            raise ValueError("Shooting policy did not provide a reference frame-0 joint position.")
        self._shooting_ref0_default_q = np.asarray(
            shooting_motion_command_0[:, : self.num_dofs], dtype=np.float32
        ).copy()
        self._recovery_ref0_default_q = self._recovery_reference_q.copy()
        self._recovery_ref0_hold_q = self._recovery_reference_q.copy()
        self.policy_action_scale = np.asarray(
            robot_configs.G1_29DOF_PER_JOINT_ACTION_SCALE, dtype=np.float32
        )
        self.goal_pos_w = np.array([config.task.goal_pos_w], dtype=np.float32)
        self._create_loco_policy(config)
        self._joint_torque_sub = JointTorqueSubscriber()
        self._wbt_executor.add_node(self._joint_torque_sub)

        logger.info(
            colored(
                "Auto-switch dual policy ready. Policy [1] is shooting; "
                "policy [2] is recovery. Recovery trigger uses policy [2] "
                "reference frame 0.",
                "green",
            )
        )

    @staticmethod
    def _json_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    def _toggle_recording(self) -> None:
        if self._recording:
            self._recording = False
            if self._record_file is not None:
                self._record_file.flush()
                self._record_file.close()
                self._record_file = None
            logger.info(f"Recording stopped: {self._record_path}")
            return

        record_dir = Path("/home/sustech/holosoma/logs/record")
        record_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._record_path = record_dir / f"dual_policy_{timestamp}.txt"
        self._record_file = self._record_path.open("w", encoding="utf-8")
        self._record_start_time = time.monotonic()
        header = {
            "type": "dual_policy_recording",
            "started_at": datetime.now().isoformat(timespec="milliseconds"),
            "shooting_policy": str(self.model_paths[0]),
            "recovery_policy": str(self.model_paths[1]),
            "locomotion_policy": str(
                Path(__file__).resolve().parent
                / "models/loco/g1_29dof/fastsac_g1_29dof.onnx"
            ),
            "rl_rate_hz": float(self.config.task.rl_rate),
            "record_format": "one JSON object per line; first line is metadata",
        }
        self._record_file.write(json.dumps(header, ensure_ascii=False) + "\n")
        self._record_file.flush()
        self._recording = True
        logger.info(f"Recording started: {self._record_path}")

    def _record_frame(self, robot_state_data: np.ndarray) -> None:
        if not self._recording or self._record_file is None or self._record_start_time is None:
            return

        try:
            self._spin_wbt_ros_once(max_callbacks=2)
        except RuntimeError as exc:
            if "context is not valid" not in str(exc):
                raise
            logger.warning("ROS2 context closed while recording; stopping the recording.")
            self._toggle_recording()
            return
        record_state = self._state
        if self.use_policy_action and self.motion_clip_progressing:
            record_state = "RECOVERY" if self.active_policy_index == 1 else "SHOOTING"
        elif self.use_policy_action and self.active_policy_index == 0:
            record_state = "SHOOT_READY"

        obs_actor = self._last_record_obs
        if obs_actor.shape[1] == 0 and record_state in {"SHOOTING", "RECOVERY", "SHOOT_READY"}:
            obs_actor = self.prepare_obs_for_rl(robot_state_data)["actor_obs"]
        policy_action_scaled = self.scaled_policy_action
        if record_state in {"SHOOT_READY", "SHOOTING", "RECOVERY"}:
            policy_action_scaled = (
                self.cmd_q.reshape(1, -1) - self._get_active_policy_default_q()
            ).astype(np.float32)
        policy_action_raw = self.last_policy_action
        if policy_action_raw.shape[1] == 0 or record_state in {"SHOOT_READY", "SHOOTING", "RECOVERY"}:
            policy_action_raw = np.divide(
                policy_action_scaled,
                self.policy_action_scale.reshape(1, -1),
                out=np.zeros_like(policy_action_scaled),
                where=self.policy_action_scale.reshape(1, -1) != 0,
            )

        if record_state == "NO_DAMPING":
            effective_kp = np.zeros(self.num_dofs, dtype=np.float32)
            effective_kd = np.zeros(self.num_dofs, dtype=np.float32)
        elif record_state == "RAMP":
            effective_kp = np.full(self.num_dofs, self.ramp_kp, dtype=np.float32)
            effective_kd = np.full(self.num_dofs, self.ramp_kd, dtype=np.float32)
        elif record_state == "STIFF":
            effective_kp = self._stiff_hold_kp
            effective_kd = self._stiff_hold_kd
        elif record_state == "LOCO" and self._loco_policy is not None:
            effective_kp = self._loco_policy.onnx_kp
            effective_kd = self._loco_policy.onnx_kd
        else:
            effective_kp = self.onnx_kp
            effective_kd = self.onnx_kd

        frame = {
            "type": "frame",
            "time_s": time.monotonic() - self._record_start_time,
            "state": record_state,
            "active_policy_index": int(self.active_policy_index),
            "motion_timestep": int(self.motion_timestep),
            "robot_state": robot_state_data[0].copy(),
            "obs_actor": obs_actor[0].copy(),
            "policy_action_raw": policy_action_raw[0].copy(),
            "policy_action_scaled": policy_action_scaled[0].copy(),
            "command_q": self.cmd_q.copy(),
            "command_dq": self.cmd_dq.copy(),
            "command_tau_ff": self.cmd_tau.copy(),
            "measured_joint_torque": self._joint_torque_sub.latest_torque.copy(),
            "object_state_pos_w": self.object_state_sub.latest_pos_w[0].copy(),
            "object_state_ori_w": self.object_state_sub.latest_ori_w[0].copy(),
            "object_state_lin_vel_w": self.object_state_sub.latest_lin_vel_w[0].copy(),
            "torso_link_pos_w": self.torso_link_pos_sub.latest_pos_w[0].copy(),
            "goal_pos_w": self.goal_pos_w[0].copy(),
            "kp": effective_kp,
            "kd": effective_kd,
        }
        self._record_file.write(json.dumps(frame, default=self._json_value, ensure_ascii=False) + "\n")

    def _record_and_spin(self, robot_state_data: np.ndarray) -> None:
        self._record_frame(robot_state_data)

    def _create_loco_policy(self, config: InferenceConfig) -> None:
        loco_path = (
            Path(__file__).resolve().parent
            / "models"
            / "loco"
            / "g1_29dof"
            #/ "fastsac_g1_29dof.onnx"
            / "ppo_g1_29dof.onnx"
        )
        if not loco_path.exists():
            raise FileNotFoundError(f"Locomotion policy not found: {loco_path}")

        class SharedInterfaceLocomotionPolicy(LocomotionPolicy):
            def _init_communication_components(self):
                self.interface = host.interface

            def _init_input_handlers(self):
                self.logger = host.logger
                self.rate = host.rate

        host = self
        loco_robot = replace(
            config.robot,
            default_dof_angles=robot_configs.g1_29dof.default_dof_angles,
            default_motor_angles=robot_configs.g1_29dof.default_motor_angles,
            motor_kp=None,
            motor_kd=None,
        )
        loco_config = InferenceConfig(
            robot=loco_robot,
            observation=observation_configs.loco_g1_29dof,
            task=replace(
                task_configs.locomotion,
                model_path=str(loco_path),
                rl_rate=config.task.rl_rate,
                domain_id=config.task.domain_id,
                interface=config.task.interface,
            ),
        )
        self._loco_policy = SharedInterfaceLocomotionPolicy(config=loco_config)
        self._loco_policy.policy_action_scale = np.asarray(
            robot_configs.G1_29DOF_PER_JOINT_ACTION_SCALE, dtype=np.float32
        )
        self.interface.robot_config = self.robot_config
        self._loco_policy.use_policy_action = True
        self._loco_policy.stand_command[0, 0] = 0
        logger.info(f"Loaded locomotion policy: {loco_path}")

    def _send_hold_command(self, robot_state_data: np.ndarray, q_target: np.ndarray, kp, kd) -> None:
        self.cmd_q[:] = q_target.reshape(-1)
        self.cmd_dq.fill(0.0)
        self.cmd_tau.fill(0.0)
        self.interface.send_low_command(
            self.cmd_q,
            self.cmd_dq,
            self.cmd_tau,
            robot_state_data[0, 7 : 7 + self.num_dofs],
            kp_override=np.asarray(kp, dtype=np.float32),
            kd_override=np.asarray(kd, dtype=np.float32),
        )

    def _validate_real_state(self, robot_state_data: np.ndarray) -> None:
        if self._deployment_mode == "sim":
            return
        expected = 7 + self.num_dofs + 6 + self.num_dofs
        if robot_state_data.shape != (1, expected) or not np.all(np.isfinite(robot_state_data)):
            raise RuntimeError(f"Invalid robot state; expected finite shape (1, {expected}).")
        quat_norm = float(np.linalg.norm(robot_state_data[0, 3:7]))
        if not 0.8 <= quat_norm <= 1.2:
            raise RuntimeError(f"Invalid IMU quaternion norm: {quat_norm:.3f}.")
        joint_speed = robot_state_data[0, 7 + self.num_dofs + 6 :]
        if np.max(np.abs(joint_speed)) > 8.0:
            raise RuntimeError("Measured joint speed exceeds the real deployment limit.")
        if not self.basketball_pose_sub.received or not self.pelvis_pose_sub.received:
            raise RuntimeError("Real deployment requires both basketball and pelvis mocap topics.")
        timeout = float(getattr(self.config.task, "mocap_timeout_seconds", 0.25))
        now = time.monotonic()
        if now - self.basketball_pose_sub.last_message_time > timeout:
            raise RuntimeError("Basketball mocap topic is missing or stale.")
        if now - self.pelvis_pose_sub.last_message_time > timeout:
            raise RuntimeError("Pelvis mocap topic is missing or stale.")

    def _guard_command(self, cmd_q, cmd_dq, cmd_tau, dof_pos_latest, kp_override, kd_override):
        q = np.asarray(cmd_q, dtype=np.float64).reshape(-1)
        dq = np.asarray(cmd_dq, dtype=np.float64).reshape(-1)
        tau = np.asarray(cmd_tau, dtype=np.float64).reshape(-1)
        if not all(np.all(np.isfinite(value)) for value in (q, dq, tau)):
            raise RuntimeError("Command contains NaN or Inf.")
        if self._deployment_mode == "sim":
            return cmd_q, cmd_dq, cmd_tau, kp_override, kd_override

        current = np.asarray(dof_pos_latest, dtype=np.float64).reshape(-1)
        if current.size != self.num_dofs or not np.all(np.isfinite(current)):
            raise RuntimeError("Measured joint position is invalid.")
        if self._deployment_mode == "armed":
            max_error = float(getattr(self.config.task, "deployment_max_target_error_rad", 0.35))
            if np.max(np.abs(q - current)) > max_error:
                raise RuntimeError("Command target is too far from measured joint position.")

        previous = current if self._last_guarded_q is None else self._last_guarded_q
        step = float(getattr(self.config.task, "deployment_max_command_step_rad", 0.02))
        guarded = previous + np.clip(q - previous, -step, step)
        self._last_guarded_q = guarded.copy()
        if self._deployment_mode == "dry-run":
            now = time.monotonic()
            if now - self._last_deployment_log >= 1.0:
                logger.info(
                    "dry-run command: q={} dq={} tau={} kp={} kd={}".format(
                        np.round(guarded, 4).tolist(),
                        np.round(dq, 4).tolist(),
                        np.round(tau, 4).tolist(),
                        None if kp_override is None else np.round(kp_override, 3).tolist(),
                        None if kd_override is None else np.round(kd_override, 3).tolist(),
                    )
                )
                self._last_deployment_log = now
        return guarded, dq, tau, kp_override, kd_override

    def _send_or_preview_command(self, robot_state_data: np.ndarray, kp_override=None, kd_override=None) -> None:
        if self._deployment_mode == "dry-run":
            now = time.monotonic()
            if now - self._last_deployment_log >= 1.0:
                logger.info(
                    "dry-run command: q={} kp={} kd={} tau={}".format(
                        np.round(self.cmd_q, 4).tolist(),
                        np.round(np.asarray(kp_override if kp_override is not None else self.onnx_kp), 3).tolist(),
                        np.round(np.asarray(kd_override if kd_override is not None else self.onnx_kd), 3).tolist(),
                        np.round(self.cmd_tau, 4).tolist(),
                    )
                )
                self._last_deployment_log = now
            return
        self.interface.send_low_command(
            self.cmd_q,
            self.cmd_dq,
            self.cmd_tau,
            robot_state_data[0, 7 : 7 + self.num_dofs],
            kp_override=kp_override,
            kd_override=kd_override,
        )

    def _start_ramp(self, robot_state_data: np.ndarray) -> None:
        self._ramp_start_q = robot_state_data[:, 7 : 7 + self.num_dofs].copy()
        self._ramp_started_at = time.monotonic()
        self._state = "RAMP"
        logger.info(
            f"Starting smooth transition to stiff_startup_pos over {self.ramp_duration_s:.1f}s "
            f"with KP={self.ramp_kp:.1f}, KD={self.ramp_kd:.1f}."
        )

    def _run_ramp_or_hold(self, robot_state_data: np.ndarray) -> None:
        if self._state == "NO_DAMPING":
            current_q = robot_state_data[:, 7 : 7 + self.num_dofs]
            self._send_hold_command(
                robot_state_data,
                current_q,
                np.zeros(self.num_dofs, dtype=np.float32),
                np.zeros(self.num_dofs, dtype=np.float32),
            )
            return

        if self._state == "RAMP":
            if self._ramp_start_q is None or self._ramp_started_at is None:
                self._start_ramp(robot_state_data)
            ramp_start_q = self._ramp_start_q
            ramp_started_at = self._ramp_started_at
            if ramp_start_q is None or ramp_started_at is None:
                raise RuntimeError("Ramp state is missing its start pose or timestamp.")
            elapsed = time.monotonic() - ramp_started_at
            ratio = min(max(elapsed / self.ramp_duration_s, 0.0), 1.0)
            smooth_ratio = 3.0 * ratio**2 - 2.0 * ratio**3
            target = ramp_start_q + (self._stiff_hold_q - ramp_start_q) * smooth_ratio
            self._send_hold_command(
                robot_state_data,
                target,
                np.full(self.num_dofs, self.ramp_kp, dtype=np.float32),
                np.full(self.num_dofs, self.ramp_kd, dtype=np.float32),
            )
            if ratio >= 1.0:
                self._state = "STIFF"
                logger.info("Smooth transition completed; stiff_startup_pos reached.")
                if self._pending_shooting_request:
                    self._pending_shooting_request = False
                    self._enter_shooting_ready()
            return

        if self._state == "STIFF":
            self._send_hold_command(robot_state_data, self._stiff_hold_q, self._stiff_hold_kp, self._stiff_hold_kd)
            if self._pending_shooting_request:
                self._pending_shooting_request = False
                self._enter_shooting_ready()

    def _switch_to_loco(self) -> None:
        if self._loco_policy is None:
            raise RuntimeError("Locomotion policy is not initialized.")
        self._auto_switch_triggered = True
        self._state = "LOCO"
        self._loco_policy.phase[:] = np.array([[0.0, np.pi]], dtype=np.float32)
        self._loco_policy.last_policy_action.fill(0.0)
        self._loco_policy.lin_vel_command.fill(0.0)
        self._loco_policy.ang_vel_command.fill(0.0)
        self._loco_policy.stand_command[0, 0] = 0
        self.use_policy_action = False
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        logger.info("Switched to locomotion policy.")

    def _enter_shooting_ready(self) -> None:
        loco_policy = self._loco_policy
        if loco_policy is None:
            raise RuntimeError("Locomotion policy is not initialized.")
        loco_policy.stand_command[0, 0] = 0
        loco_policy.use_policy_action = False
        self._activate_policy(0)
        self.force_zero_obj_pos_b = False
        self.goal_pos_w = np.array([self.config.task.goal_pos_w], dtype=np.float32)
        self._handle_start_policy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self._state = "SHOOT_READY"
        logger.info("Shooting policy ready at reference frame 0. Press s to start the motion clip.")

    def _start_shooting(self) -> None:
        self._handle_start_motion_clip()
        self._state = "SHOOTING"
        self._auto_switch_triggered = False
        logger.info("Shooting motion clip started.")

    def handle_keyboard_button(self, keycode):
        key = str(keycode).lower()
        if key in {"enter", "return", "\n", "\r"} and self._state == "NO_DAMPING":
            state = self.interface.get_low_state()
            self._start_ramp(state)
        elif key == "l" and self._state == "STIFF":
            self._switch_to_loco()
        elif key == "]":
            if self._state == "STIFF":
                self._enter_shooting_ready()
            elif self._state == "RAMP":
                self._pending_shooting_request = True
                logger.info("Shooting request queued until stiff_startup_pos is reached.")
            elif self._state == "LOCO":
                self._enter_shooting_ready()
        elif key == "s" and self._state == "SHOOT_READY":
            self._start_shooting()
        elif key == "r":
            self._toggle_recording()
        elif key == "o":
            self._state = "STIFF"
            self._auto_switch_triggered = False
            self._handle_stop_policy()
        elif key in {"1", "2"}:
            super().handle_keyboard_button(key)
        else:
            if self._state == "LOCO" and self._loco_policy is not None:
                self._loco_policy.handle_keyboard_button(key)
            else:
                super().handle_keyboard_button(key)

    def policy_action(self):
        robot_state_data = self.interface.get_low_state()
        if self._deployment_mode != "sim":
            self._spin_wbt_ros_once(max_callbacks=4)
            self._validate_real_state(robot_state_data)
        if self._state in {"NO_DAMPING", "RAMP", "STIFF"}:
            self._run_ramp_or_hold(robot_state_data)
            self._record_and_spin(robot_state_data)
            return

        if self._state == "LOCO":
            loco_policy = self._loco_policy
            if loco_policy is None:
                raise RuntimeError("Locomotion policy is not initialized.")
            loco_action = loco_policy.rl_inference(robot_state_data)
            self._last_record_obs = loco_policy.prepare_obs_for_rl(robot_state_data)["actor_obs"].copy()
            q_target = loco_action + loco_policy.default_dof_angles
            self._send_hold_command(
                robot_state_data,
                q_target,
                loco_policy.onnx_kp,
                loco_policy.onnx_kd,
            )
            self._record_and_spin(robot_state_data)
            return

        super().policy_action()
        if self.active_policy_index == 1 and self.motion_timestep >= self.recovery_end_frame:
            self._switch_to_loco()

        self._record_and_spin(robot_state_data)

    def _recovery_pose_matches(self, robot_state_data: np.ndarray) -> bool:
        if self.motion_timestep < self.recovery_min_shooting_steps:
            return False

        current_q = np.asarray(
            robot_state_data[:, 7 : 7 + self.num_dofs], dtype=np.float32
        ).reshape(1, -1)
        abs_error = np.abs(current_q - self._recovery_reference_q)
        mean_abs_error = float(np.mean(abs_error))
        max_abs_error = float(np.max(abs_error))
        matched = (
            mean_abs_error <= self.recovery_mean_abs_threshold
            and max_abs_error <= self.recovery_max_abs_threshold
        )

        if matched:
            logger.info(
                "Recovery pose matched: mean_abs_error={:.4f}, max_abs_error={:.4f}".format(
                    mean_abs_error, max_abs_error
                )
            )
        return matched

    def _switch_to_recovery(self) -> None:
        if len(self.model_paths) < 2:
            raise RuntimeError("Recovery policy is not loaded.")

        self._auto_switch_triggered = True
        self._reset_obs_history_buffers()
        self._activate_policy(1)

        self.force_zero_obj_pos_b = True
        self._zero_obj_pos_b_debug_printed = False
        self.goal_pos_w = np.zeros((1, 3), dtype=np.float32)

        self._handle_start_policy()
        self._handle_start_motion_clip()
        self._state = "RECOVERY"
        logger.info(
            colored(
                "Automatically switched to recovery policy. obj_pos_b=[0, 0, 0], "
                "goal_pos_w=[0, 0, 0], recovery tracking started from reference "
                "frame 0.",
                "green",
            )
        )

    def _infer_active_wbt_policy(self, robot_state_data: np.ndarray):
        if not self.motion_clip_progressing:
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None

        obs = self.prepare_obs_for_rl(robot_state_data)
        self._last_record_obs = obs["actor_obs"].copy()
        input_feed = {
            "time_step": np.array([[self.motion_timestep]], dtype=np.float32),
            "obs": obs["actor_obs"],
        }
        policy_action, self.motion_command_t, self.ref_quat_xyzw_t = self.policy(input_feed)
        policy_action = np.clip(policy_action, -100, 100)
        self.last_policy_action = policy_action.copy()
        self.scaled_policy_action = policy_action * self.policy_action_scale

        if self.motion_clip_progressing:
            if self.use_sim_time:
                self._update_clock()
            else:
                self.motion_timestep += 1

        return self.scaled_policy_action

    def rl_inference(self, robot_state_data: np.ndarray):
        scaled_action = self._infer_active_wbt_policy(robot_state_data)

        if (
            not self._auto_switch_triggered
            and self.active_policy_index == 0
            and self.motion_clip_progressing
            and self.use_policy_action
            and self._recovery_pose_matches(robot_state_data)
        ):
            self._switch_to_recovery()
            return self._infer_active_wbt_policy(robot_state_data)
        return scaled_action

    def get_current_obs_buffer_dict(self, robot_state_data: np.ndarray):
        observations = super().get_current_obs_buffer_dict(robot_state_data)
        observations["goal_pos_w"] = (
            np.zeros((1, 3), dtype=np.float32)
            if self.active_policy_index == 1 and self.force_zero_obj_pos_b
            else self.goal_pos_w
        )
        return observations


def _print_control_guide(use_joystick: bool) -> None:
    logger.info("=" * 80)
    logger.info("AUTO-SWITCH DUAL WBT POLICY CONTROLS")
    logger.info("=" * 80)
    if use_joystick:
        logger.info("A button       - Start current policy")
        logger.info("B button       - Stop policy")
        logger.info("Y button       - Set robot to default pose")
        logger.info("Start button   - Start current motion clip")
        logger.info("Select button  - Manually cycle policy slot")
    else:
        logger.info("Enter          - Smoothly transition from no damping to stiff startup pose")
        logger.info("]              - Prepare shooting policy after stiff startup")
        logger.info("s              - Start shooting motion clip")
        logger.info("r              - Start/stop recording to logs/record/*.txt")
        logger.info("o              - Stop policy")
    logger.info("Recovery switches automatically at shooting completion, then returns to loco at frame 80.")
    logger.info("=" * 80)


def run_policy(config: InferenceConfig) -> None:
    logger.info("Starting automatic dual WBT policy...")
    logger.info(f"Robot: {config.robot.robot_type}")
    logger.info(f"Observation groups: {list(config.observation.obs_dict.keys())}")
    logger.info(f"RL rate: {config.task.rl_rate} Hz")
    logger.info(f"Model path(s): {config.task.model_path}")

    try:
        policy = AutoSwitchDualPolicy(config=config)
        logger.info("Policy initialized successfully!")
        _print_control_guide(config.task.use_joystick)
        policy.run()
    except Exception as exc:
        logger.error(f"Error running automatic dual policy: {exc}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        restore_terminal_settings()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--deploy-mode", choices=("sim", "read-only", "dry-run", "armed"), default="sim")
    known, remaining = parser.parse_known_args(sys.argv[1:])
    if known.deploy_mode == "read-only" and not any(
        argument == "--task.model-path" or argument.startswith("--task.model-path=")
        for argument in remaining
    ):
        remaining.extend(["--task.model-path", "__read_only__"])
    sys.argv = [sys.argv[0], *remaining]
    config = tyro.cli(AnnotatedInferenceConfig, config=TYRO_CONFIG)
    if known.deploy_mode in {"read-only", "dry-run", "armed"}:
        if config.task.interface in {None, "", "lo"}:
            raise ValueError("Real deployment requires the physical robot network interface, not lo.")
        if config.task.use_sim_time:
            raise ValueError("Simulation time is forbidden in real deployment modes.")
        if known.deploy_mode == "armed":
            if not sys.stdin.isatty():
                raise RuntimeError("Armed deployment requires an interactive terminal.")
            acknowledgement = input("Type I_UNDERSTAND_THIS_COMMANDS_A_REAL_ROBOT to continue: ").strip()
            if acknowledgement != "I_UNDERSTAND_THIS_COMMANDS_A_REAL_ROBOT":
                raise RuntimeError("Armed deployment acknowledgement did not match.")
        config = replace(
            config,
            task=replace(
                config.task,
                deployment_mode=known.deploy_mode,
                deployment_write_enabled=known.deploy_mode == "armed",
                deployment_configure_control_mode=known.deploy_mode == "armed",
            ),
        )
        if known.deploy_mode == "read-only":
            ReadOnlyRobotMonitor(config).run()
            return
    run_policy(config)


if __name__ == "__main__":
    main()
