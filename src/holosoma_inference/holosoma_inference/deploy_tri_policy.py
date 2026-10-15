#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import os
import select
import termios
import tty
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(PACKAGE_ROOT.parent), str(PACKAGE_ROOT.parents[1] / "holosoma")]

import numpy as np
import rclpy
import tyro
from geometry_msgs.msg import PoseStamped
from loguru import logger
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from holosoma_inference.config.config_values.inference import AnnotatedInferenceConfig
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.motion_paths import MOTIONS_DIR, resolve_motion_path
from holosoma_inference.policies.wbt_two_policy import PinocchioRobot
from holosoma_inference.run_three_policy import ThreePolicy
from holosoma_inference.sdk.interface_wrapper import InterfaceWrapper
from holosoma_inference.utils.math.quat import wxyz_to_xyzw, xyzw_to_wxyz, matrix_from_quat
from holosoma_inference.utils.misc import restore_terminal_settings
from holosoma_inference.utils.rate import RateLimiter
from defusedxml import ElementTree


DEFAULT_REFERENCE = str(MOTIONS_DIR / "basketball_013_pick_extend_ball_fixed_new.npz")


@dataclass(frozen=True)
class Options:
    mode: str = "read-only"
    pelvis_topic: str = "/pelvis/pose"
    ball_topic: str = "/basketball/pose"
    mocap_unit: str = "mm"
    reference: str = DEFAULT_REFERENCE
    tolerance: float = 0.03
    stable_seconds: float = 0.5
    sensor_timeout: float = 0.25
    max_stamp_skew: float = 0.04
    startup_pose_error: float = 0.10
    startup_speed: float = 0.3
    ramp_seconds: float = 10.0
    max_command_step: float = 0.02
    max_target_error: float = 0.35
    max_joint_speed: float = 8.0
    max_read_seconds: float = 0.1
    exit_damping_kd: float = 2.0
    loco_action_scale: str = ""
    loco_urdf: str = str(PACKAGE_ROOT.parents[1] / "holosoma/holosoma/data/robots/g1/g1_29dof.urdf")
    max_iterations: int = 0
    record: bool = False


class RobotInterface(InterfaceWrapper):
    command_closed = False
    armed_capability = False

    def send_low_command(self, *args, **kwargs):
        if self.command_closed or (self.write_enabled and not self.armed_capability):
            raise RuntimeError("Robot command channel is closed or lacks armed capability.")
        return super().send_low_command(*args, **kwargs)

    def _send_binding_command(self, *args, **kwargs):
        if self.command_closed or not self.write_enabled or not self.armed_capability:
            raise RuntimeError("Native write boundary rejects non-armed or closed commands.")
        return super()._send_binding_command(*args, **kwargs)
    def _convert_binding_state_to_array(self):
        raw = self.unitree_interface.read_low_state()
        tick = int(raw.tick)
        previous = getattr(self, "last_tick", None)
        if tick != previous:
            forward = previous is not None and (tick > previous or (previous > 2**32 - 1024 and tick < 1024))
            self.tick_advances = getattr(self, "tick_advances", 0) + 1 if forward else 0
            self.last_tick = tick
            self.last_tick_at = time.monotonic()
        q = np.asarray(raw.motor.q)[list(self.robot_config.joint2motor)]
        dq = np.asarray(raw.motor.dq)[list(self.robot_config.joint2motor)]
        tau = getattr(raw.motor, "tau_est", None)
        self.motor_tau_est = None if tau is None else np.asarray(tau)[list(self.robot_config.joint2motor)].copy()
        if self.motor_tau_est is not None and (self.motor_tau_est.shape != (29,) or not np.all(np.isfinite(self.motor_tau_est))):
            self.motor_tau_est = None
        return np.concatenate([np.zeros(3), np.asarray(raw.imu.quat), q, np.zeros(3),
                               np.asarray(raw.imu.omega), dq]).reshape(1, -1)

    def require_fresh_state(self, timeout):
        if getattr(self, "last_tick", 0) <= 0 or time.monotonic() - getattr(self, "last_tick_at", 0) > timeout:
            raise RuntimeError("Unitree DDS lowstate tick 未更新或已过期")
        if getattr(self, "tick_advances", 0) < 1:
            raise RuntimeError("尚未观察到 Unitree DDS lowstate tick 连续推进")

    def await_state(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.get_low_state()
            try:
                self.require_fresh_state(0.25)
                if state.shape == (1, 71) and np.all(np.isfinite(state)) and 0.8 <= np.linalg.norm(state[0, 3:7]) <= 1.2:
                    return
            except RuntimeError:
                pass
            time.sleep(0.002)
        raise RuntimeError("初始化未收到持续更新的有效 Unitree lowstate。")


class TrackedPose(Node):
    def __init__(self, topic, name, unit, orientation_required=True):
        super().__init__(name)
        self.scale = {"mm": 0.001, "cm": 0.01, "m": 1.0}[unit]
        self.sample = None
        self.orientation_required = orientation_required
        self.last_invalid = None
        self.create_subscription(PoseStamped, topic, self._callback, 1)

    def _callback(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        raw = np.array([p.x, p.y, p.z], dtype=np.float64)
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        norm = np.linalg.norm(quat)
        if not np.all(np.isfinite(raw)) or (self.orientation_required and (not np.all(np.isfinite(quat)) or not 0.8 <= norm <= 1.2)):
            self.sample = None
            self.last_invalid = "动捕位置或四元数无效"
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.sample is not None and stamp <= self.sample["stamp"]:
            self.last_invalid = "动捕源时间戳未推进"
            return
        normalized = quat / norm if np.all(np.isfinite(quat)) and norm > 1e-8 else np.array([0., 0., 0., 1.])
        self.sample = {"raw_position": raw, "raw_quat_xyzw": quat.copy() if np.all(np.isfinite(quat)) else None,
                       "position": raw * self.scale, "quat_xyzw": normalized,
                       "received_at": time.monotonic(), "frame_id": msg.header.frame_id,
                       "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9}
        self.last_invalid = None


class Tracking:
    def __init__(self, options):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.options = options
        self._closed = False
        self.pelvis = TrackedPose(options.pelvis_topic, "tri_pelvis_tracker", options.mocap_unit)
        self.ball = TrackedPose(options.ball_topic, "tri_ball_tracker", options.mocap_unit, orientation_required=False)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.pelvis)
        self.executor.add_node(self.ball)

    def refresh(self):
        for _ in range(8):
            self.executor.spin_once(timeout_sec=0.0)

    def require(self, ball=False):
        now = time.monotonic()
        for label, node in (("pelvis", self.pelvis), ("ball", self.ball)) if ball else (("pelvis", self.pelvis),):
            if node.sample is None or now - node.sample["received_at"] > self.options.sensor_timeout:
                raise RuntimeError(f"{label} 动捕未收到、无效或过期")
            source_age = node.get_clock().now().nanoseconds * 1e-9 - node.sample["stamp"]
            if not -self.options.max_stamp_skew <= source_age <= self.options.sensor_timeout:
                raise RuntimeError(f"{label} 动捕源时间戳过期或与本机时钟不一致")
        if ball:
            p, b = self.pelvis.sample, self.ball.sample
            if not p["frame_id"] or p["frame_id"] != b["frame_id"]:
                raise RuntimeError("pelvis 与篮球的动捕 global frame_id 不一致或为空")
            if min(p["stamp"], b["stamp"]) <= 0 or abs(p["stamp"] - b["stamp"]) > self.options.max_stamp_skew:
                raise RuntimeError("pelvis 与篮球动捕时间戳无效或不同步")

    def torso_pose(self, state, robot):
        self.require()
        return torso_from_pelvis(state, robot, self.pelvis.sample)

    def close(self):
        if self._closed:
            return
        errors = []
        if self.executor is not None:
            try:
                self.executor.shutdown()
                self.executor = None
            except Exception as exc:
                errors.append(exc)
        for name in ("pelvis", "ball"):
            node = getattr(self, name)
            if node is not None:
                try:
                    node.destroy_node()
                    setattr(self, name, None)
                except Exception as exc:
                    errors.append(exc)
        self._closed = not errors
        if errors:
            raise RuntimeError(f"Tracking cleanup incomplete: {errors}")


def torso_from_pelvis(state, robot, pose):
    joints = state[0, 7:36][robot.real2pinocchio_index]
    configuration = np.concatenate([pose["position"], pose["quat_xyzw"], joints])
    pos, quat = robot.fk_and_get_ref_body_pose_in_world(configuration)
    return pos.copy(), xyzw_to_wxyz(quat)


def reference_ball_position(path):
    with np.load(path, allow_pickle=False) as data:
        index = data["body_names"].tolist().index("torso_link")
        pos = data["body_pos_w"][0, index]
        quat = data["body_quat_w"][0, index]
        if not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1e-8:
            raise ValueError("Invalid reference torso quaternion.")
        rotation = matrix_from_quat((quat / np.linalg.norm(quat)).reshape(1, 4))[0]
        target = rotation.T @ (data["object_pos_w"][0] - pos)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Invalid reference basketball position.")
    return target


def validate_state(state, seconds, options):
    if state is None or state.shape != (1, 71) or not np.all(np.isfinite(state)):
        raise RuntimeError("Unitree lowstate 必须是有效的 (1,71) 数据")
    if seconds > options.max_read_seconds:
        raise RuntimeError("Unitree lowstate 读取耗时超限")
    if not 0.8 <= np.linalg.norm(state[0, 3:7]) <= 1.2:
        raise RuntimeError("IMU 四元数无效")
    if np.max(np.abs(state[0, 42:71])) > options.max_joint_speed:
        raise RuntimeError("实测关节速度超限")


class PlacementGate:
    def __init__(self, options, target):
        self.options, self.target = options, target
        self.valid_since = None
        self.last_report = -float("inf")
        self.last_check = None
        self.snapshot = {"ok": False, "reason": "等待数据"}

    def check(self, actual, pose_ok, sensor_error=None):
        now = time.monotonic()
        if self.last_check is not None and now - self.last_check > self.options.sensor_timeout:
            self.valid_since = None
        self.last_check = now
        delta = None if actual is None else self.target - actual
        good = sensor_error is None and pose_ok and delta is not None and bool(np.all(np.abs(delta) <= self.options.tolerance))
        self.valid_since = (now if self.valid_since is None else self.valid_since) if good else None
        ready = good and now - self.valid_since >= self.options.stable_seconds
        self.snapshot = {"ok": bool(ready), "actual_obj_pos_b": actual, "target_obj_pos_b": self.target,
                         "move_in_torso": delta, "pose_ok": bool(pose_ok), "reason": sensor_error}
        if now - self.last_report >= 1:
            if ready:
                logger.info("篮球位置预检：行；read-only 不开启 policy，需在 dry-run/armed 再确认。" if self.options.mode == "read-only"
                            else "篮球放置：行，可以按 ] 开启 pick policy。")
            elif sensor_error:
                logger.warning("篮球放置：不行，{}", sensor_error)
            elif not pose_ok:
                logger.warning("篮球放置：不行，机器人尚未达到稳定的 pick 初始姿态。")
            elif good:
                logger.info("篮球位置在范围内，等待稳定 {:.1f} 秒。", self.options.stable_seconds)
            else:
                logger.warning("篮球放置：不行，沿 torso 坐标轴移动 x{:+.1f}cm y{:+.1f}cm z{:+.1f}cm。容差每轴 ±{:.1f}cm",
                               *(delta * 100), self.options.tolerance * 100)
            self.last_report = now
        return ready


class TriDeployment(ThreePolicy):
    def __init__(self, config, options):
        self.options = options
        self._initializing = True
        self._fault = None
        self._tracking_closed = False
        self._arm_requested = False
        self._startup_complete = False
        self._holding_loco = False
        self._keyboard_fd = None
        self._terminal_settings = None
        self.tracking = Tracking(options)
        try:
            super().__init__(config)
            self._finish_initialization()
        except Exception:
            self.close()
            raise

    def _finish_initialization(self):
        options = self.options
        self.interface.armed_capability = options.mode == "armed"
        self._configure_loco_contract()
        self.ramp_duration_s = options.ramp_seconds
        self._stiff_hold_q = self._policy_states[0]["contract"]["default_q"].copy()
        metadata = self._policy_states[0]["onnx_policy_session"].get_modelmeta().custom_metadata_map
        experiment = json.loads(metadata["experiment_config"])
        motion_path = experiment["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]["motion_file"]
        self.gate = PlacementGate(options, reference_ball_position(resolve_motion_path(motion_path)))
        self._last_real_state = None
        self._last_real_state_at = 0.0
        self._last_preview_at = -float("inf")
        self._last_torso_pose = None
        self._initializing = False
        if options.record:
            self._toggle_recording()
        if sys.stdin.isatty():
            self._init_keyboard_handler()
        logger.info("{}: Enter 缓升到 pick default pose；摆球；位置合格后 ]；s 开始 tracking；o 保持；r 记录。", options.mode)

    def _validate_environment(self, config):
        validate_config(config, self.options)

    def setup_policy(self, model_path):
        super().setup_policy(model_path)
        metadata = self.onnx_policy_session.get_modelmeta().custom_metadata_map
        robot = json.loads(metadata["experiment_config"])["robot"]
        effort = np.asarray(robot["dof_effort_limit_list"], dtype=np.float64)
        if effort.shape != (29,) or not np.all(np.isfinite(effort)) or np.any(effort <= 0):
            raise ValueError("Invalid model effort limits.")
        self._model_contract["effort_limits"] = effort

    def _init_keyboard_handler(self):
        if not self._initializing:
            if not sys.stdin.isatty():
                return
            self._keyboard_fd = sys.stdin.fileno()
            self._terminal_settings = termios.tcgetattr(self._keyboard_fd)
            tty.setcbreak(self._keyboard_fd)

    def _poll_keyboard(self):
        if self._closed or self._keyboard_fd is None:
            return
        for _ in range(16):
            if not select.select([self._keyboard_fd], [], [], 0)[0]:
                break
            value = os.read(self._keyboard_fd, 1)
            if not value or value in {b"\x1b", b"\x04"}:
                self.close()
                break
            self._keyboard_press(value.decode("utf-8", errors="ignore"))
            if self._closed:
                break

    def _keyboard_press(self, key):
        try:
            self.handle_keyboard_button(key)
        except Exception as exc:
            with self._control_lock:
                self._fault = str(exc)
                self.close()
            logger.exception("键盘指令失败，已停止部署控制。")

    def _init_communication_components(self):
        self.interface = RobotInterface(self.robot_config, self.config.task.domain_id,
            self.config.task.interface, False, configure_control_mode=False, write_enabled=False)
        self.interface.await_state()

    def _configure_loco_contract(self):
        loco = self._loco_policy
        metadata = loco.onnx_policy_session.get_modelmeta().custom_metadata_map
        if json.loads(metadata["dof_names"]) != list(self.robot_config.dof_names):
            raise ValueError("Loco model joint order mismatch.")
        raw = metadata.get("action_scale") or self.options.loco_action_scale
        if not raw:
            raise ValueError("Loco ONNX has no action_scale metadata; supply --deploy-loco-action-scale explicitly.")
        try:
            scale = np.asarray(json.loads(raw), dtype=np.float32).reshape(-1)
        except json.JSONDecodeError:
            scale = np.asarray([float(x) for x in raw.split(",")], dtype=np.float32)
        if scale.size == 1:
            scale = np.full(29, scale.item(), dtype=np.float32)
        if scale.shape != (29,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("Loco scale must have one or 29 positive finite values.")
        text = Path(self.options.loco_urdf).read_text()
        joints = {j.attrib["name"]: j for j in ElementTree.fromstring(text).findall("joint")}
        efforts = np.asarray([float(joints[name].find("limit").attrib["effort"]) for name in self.dof_names])
        self._loco_contract = {"robot": PinocchioRobot(loco.robot_config, text), "effort_limits": efforts}
        for array in (loco.onnx_kp, loco.onnx_kd, loco.default_dof_angles.reshape(-1), efforts):
            if np.asarray(array).shape != (29,) or not np.all(np.isfinite(array)):
                raise ValueError("Invalid loco contract.")
        if np.any(loco.onnx_kp <= 0) or np.any(loco.onnx_kd < 0) or np.any(efforts <= 0):
            raise ValueError("Invalid loco gains/efforts.")
        loco.policy_action_scale = scale
        if loco.onnx_policy_session.get_inputs()[0].shape != [1, 100]:
            raise ValueError("Expected the supported 100-dimensional loco observation contract.")

    def _get_ref_body_pose_in_world(self, state):
        pos, quat = self.tracking.torso_pose(state, self.pinocchio_robot)
        self._last_torso_pose = {"position": pos, "quat_wxyz": quat}
        return pos, quat

    def _run_ramp_or_hold(self, state):
        if self._state != "RAMP":
            return super()._run_ramp_or_hold(state)
        ratio = np.clip((time.monotonic() - self._ramp_started_at) / self.ramp_duration_s, 0, 1)
        alpha = ratio * ratio * (3 - 2 * ratio)
        target = self._ramp_start_q + alpha * (self._stiff_hold_q - self._ramp_start_q)
        self._send_hold_command(state, target, self.onnx_kp, self.onnx_kd)
        if ratio >= 1:
            self._state = "STIFF"
            self._startup_complete = True
            logger.info("初始姿态过渡完成；请放球并等待位置检查通过后按 ]。")

    def _placement_ready(self, state):
        actual, error = None, None
        try:
            self.tracking.require(ball=True)
            pos, quat = self._get_ref_body_pose_in_world(state)
            actual = matrix_from_quat(quat)[0].T @ (self.tracking.ball.sample["position"] - pos[0])
        except RuntimeError as exc:
            error = str(exc)
        initial = self._policy_states[0]["contract"]["default_q"][0]
        pose_ok = (np.max(np.abs(state[0, 7:36] - initial)) <= self.options.startup_pose_error
                   and np.max(np.abs(state[0, 42:71])) <= self.options.startup_speed)
        return self.gate.check(actual, pose_ok, error)

    def handle_keyboard_button(self, keycode):
        with self._control_lock:
            if self._closed or self._fault:
                return
            key = str(keycode).lower()
            if key in {"v", "b", "f", "g"}:
                logger.warning("真机入口禁用临时增益快捷键。")
                return
            if key == "o" and self._state == "NO_DAMPING":
                return
            if self._state == "LOCO" and key in {"w", "a", "s", "d", "q", "e", "z", "="}:
                super().handle_keyboard_button(key)
                return
            if key in {"enter", "return", "\n", "\r", "]", "s", "o"}:
                self.tracking.refresh()
                started = time.monotonic()
                state = self.interface.get_low_state()
                try:
                    validate_state(state, time.monotonic() - started, self.options)
                    self.interface.require_fresh_state(self.options.sensor_timeout)
                    self.tracking.require()
                except RuntimeError as exc:
                    if not self.interface.write_enabled:
                        logger.warning("拒绝指令，继续等待有效状态/动捕：{}", exc)
                        return
                    raise
            if key in {"enter", "return", "\n", "\r"}:
                if self._state not in {"NO_DAMPING", "STIFF"}:
                    return
                self._activate_policy(0)
                self._arm_requested = True
                self._startup_complete = False
                self._holding_loco = False
                self._stiff_hold_q = self._get_active_policy_default_q().copy()
                self._stiff_hold_kp, self._stiff_hold_kd = self.onnx_kp.copy(), self.onnx_kd.copy()
                self._pending_shooting_request = False
                self._start_ramp(state)
                return
            if key == "]":
                if not self._startup_complete or self._state != "STIFF" or not self._placement_ready(state):
                    logger.warning("拒绝开启 policy：先完成 Enter 缓升并确保篮球放置检查通过。")
                    return
                self._holding_loco = False
            if key == "s" and (self._state != "PICK_READY" or not self._placement_ready(state)):
                logger.warning("拒绝 tracking：pick 未就绪或篮球放置检查未通过。")
                return
            if key == "o":
                self._holding_loco = self._state == "LOCO" or (self._state == "STIFF" and self._holding_loco)
                self._pending_shooting_request = False
                self.use_policy_action = False
                self.motion_clip_progressing = False
                self._stiff_hold_q = state[:, 7:36].copy()
                model = self._loco_policy if self._holding_loco else self
                self._stiff_hold_kp, self._stiff_hold_kd = model.onnx_kp.copy(), model.onnx_kd.copy()
                self._state = "STIFF"
                return
            super().handle_keyboard_button(key)

    def get_current_obs_buffer_dict(self, state):
        self.tracking.require(ball=self.active_policy_index != 2)
        pos, quat = self._get_ref_body_pose_in_world(state)
        self.object_state_sub.latest_pos_w[:] = self.tracking.ball.sample["position"] if self.tracking.ball.sample else 0
        observations = self._robot_observations(state, quat)
        observations["obj_pos_b"] = (np.zeros((1, 3), dtype=np.float32) if self.force_zero_obj_pos_b else
            self._obj_pos_in_ref_frame(self.object_state_sub.latest_pos_w, pos, quat))
        observations["goal_pos_w"] = self.goal_pos_w
        return observations

    def _capture_command(self, q, dq, tau, measured, kp, kd):
        if self._closed or self._fault or not self._arm_requested:
            raise RuntimeError("Command rejected: closed/faulted or Enter has not authorized startup.")
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        current = np.asarray(measured, dtype=np.float64).reshape(-1)
        loco_contract = self._state == "LOCO" or self._holding_loco
        if loco_contract:
            limits_robot = self._loco_contract["robot"]
        else:
            limits_robot = self.pinocchio_robot
        lower = np.empty(29)
        upper = np.empty(29)
        indices = limits_robot.real2pinocchio_index
        lower[indices] = limits_robot.robot_model.lowerPositionLimit[7:]
        upper[indices] = limits_robot.robot_model.upperPositionLimit[7:]
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower >= upper):
            raise RuntimeError("Invalid mapped URDF limits.")
        if q.shape != (29,) or current.shape != (29,) or not np.all(np.isfinite(q)):
            raise RuntimeError("Invalid command shape or non-finite target.")
        if np.any(q < lower) or np.any(q > upper):
            raise RuntimeError("Command exceeds URDF joint limits.")
        if np.any(current < lower - 1e-5) or np.any(current > upper + 1e-5):
            raise RuntimeError("Measured starting pose is outside URDF limits.")
        previous = current if self._last_guarded_q is None else self._last_guarded_q
        bounded = previous + np.clip(q - previous, -self.options.max_command_step, self.options.max_command_step)
        if np.any(bounded < lower) or np.any(bounded > upper):
            raise RuntimeError("Bounded command exceeds URDF joint limits.")
        if np.max(np.abs(bounded - current)) > self.options.max_target_error:
            raise RuntimeError("Bounded target error exceeds the measured-pose limit.")
        result = super()._capture_command(bounded, dq, tau, measured, kp, kd)
        estimated = self._last_command["kp"] * (bounded - current)
        estimated += self._last_command["kd"] * (np.asarray(dq) - self._last_real_state[0, 42:71]) + np.asarray(tau)
        efforts = self._loco_contract["effort_limits"] if loco_contract else self._model_contract["effort_limits"]
        if not np.all(np.isfinite(estimated)) or np.any(np.abs(estimated) > efforts):
            raise RuntimeError("Predicted PD torque exceeds a joint's model effort limit.")
        self._last_guarded_q = bounded.copy()
        if self.options.mode == "armed":
            self.interface.require_fresh_state(self.options.sensor_timeout)
            self.tracking.require(ball=self._state in {"PICK_READY", "PICKING", "SHOOTING"})
            if self._last_real_state is None or time.monotonic() - self._last_real_state_at > self.options.sensor_timeout:
                raise RuntimeError("No recent validated robot-state sample.")
            if not self.interface.write_enabled:
                import unitree_interface
                self.interface.unitree_interface.set_control_mode(unitree_interface.ControlMode.PR)
                self.interface.write_enabled = True
        elif self.interface.write_enabled:
            raise RuntimeError("Non-armed communication write gate must remain closed.")
        if time.monotonic() - self._last_preview_at >= 1:
            logger.info("{} command preview: q={} kp={} kd={} PD估计={}；write_enabled={}", self.options.mode,
                        np.round(bounded, 4).tolist(), self._last_command["kp"].tolist(), self._last_command["kd"].tolist(),
                        np.round(estimated, 3).tolist(), self.interface.write_enabled)
            self._last_preview_at = time.monotonic()
        return result

    def policy_action(self):
        with self._control_lock:
            if self._fault or self._closed:
                return
            self.tracking.refresh()
            started = time.monotonic()
            state = self.interface.get_low_state()
            validate_state(state, time.monotonic() - started, self.options)
            self.interface.require_fresh_state(self.options.sensor_timeout)
            self._last_real_state, self._last_real_state_at = state, time.monotonic()
            self._joint_torque_sub.received = self.interface.motor_tau_est is not None
            if self._joint_torque_sub.received:
                self._joint_torque_sub.latest_torque = self.interface.motor_tau_est.copy()
            if self._state in {"NO_DAMPING", "RAMP", "STIFF", "PICK_READY"}:
                self._placement_ready(state)
            if self._state == "NO_DAMPING":
                self._record_frame(state)
                return
            self.tracking.require(ball=self._state in {"PICK_READY", "PICKING", "SHOOTING"})
            if self._state in {"RAMP", "STIFF"}:
                self._run_ramp_or_hold(state)
            elif self._state == "LOCO":
                self._loco_policy.update_phase_time()
                action = self._loco_policy.rl_inference(state)
                self._send_hold_command(state, action + self._loco_policy.default_dof_angles,
                                        self._loco_policy.onnx_kp, self._loco_policy.onnx_kd)
            else:
                action = self.rl_inference(state)
                self._send_hold_command(state, action + self._get_active_policy_default_q(), self.onnx_kp, self.onnx_kd)
            self._record_frame(state)
            if self._state == "RECOVERY" and self.motion_timestep >= self.recovery_end_frame:
                self._switch_to_loco()

    def _record_frame(self, state):
        super()._record_frame(state)
        if self._recording:
            self._record_file.write(json.dumps({"type": "tracking", "mode": self.options.mode,
                "placement": self.gate.snapshot, "pelvis": self.tracking.pelvis.sample,
                "ball": self.tracking.ball.sample, "mocap_unit": self.options.mocap_unit,
                "torso": self._last_torso_pose, "robot_yaw_offset": self.robot_yaw_offset,
                "motion_yaw_offset": self.motion_yaw_offset,
                "lowstate_tick": self.interface.last_tick, "torque_source": "unitree_lowstate_tau_est",
                "ball_velocity_available": False,
                "commands_enabled": self.interface.write_enabled},
                default=self._json_value) + "\n")

    def _stop_recording(self):
        self._recording = False
        handle = self._record_file
        if handle is not None:
            try:
                handle.flush()
            finally:
                try:
                    handle.close()
                finally:
                    if handle.closed:
                        self._record_file = None

    def run(self):
        try:
            iteration = 0
            while not self._closed:
                self._poll_keyboard()
                if self._closed:
                    break
                self.policy_action()
                iteration += 1
                if self.options.max_iterations and iteration >= self.options.max_iterations:
                    break
                self.rate.sleep()
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            with self._control_lock:
                self._fault = str(exc)
                self.use_policy_action = False
                self.motion_clip_progressing = False
                self.close()
            logger.exception("Deployment fault; policy stopped and further command writes disabled.")
            raise
        finally:
            self.close()

    def close(self):
        lock = getattr(self, "_control_lock", None)
        if lock is None:
            if hasattr(self, "tracking"):
                self.tracking.close()
            return
        with lock:
            interface = getattr(self, "interface", None)
            try:
                if not self._closed and interface is not None and self.options.mode == "armed" and interface.write_enabled:
                    guard = self.interface.command_guard
                    try:
                        self.interface.command_guard = None
                        started = time.monotonic()
                        state = self.interface.get_low_state()
                        valid = state is not None and state.shape == (1, 71) and np.all(np.isfinite(state)) and time.monotonic() - started <= self.options.max_read_seconds
                        try:
                            self.interface.require_fresh_state(self.options.sensor_timeout)
                        except RuntimeError:
                            valid = False
                        if not valid:
                            logger.warning("退出未取得新鲜 lowstate：请求零增益而不是根据旧速度估计阻尼。")
                        q = state[0, 7:36] if valid else np.zeros(29)
                        efforts = self._loco_contract["effort_limits"] if hasattr(self, "_loco_contract") else self._model_contract["effort_limits"]
                        kd = np.minimum(self.options.exit_damping_kd, efforts / np.maximum(np.abs(state[0, 42:71]), 1e-8)) if valid else np.zeros(29)
                        level = self.interface.kd_level
                        if not np.isfinite(level) or level <= 0:
                            raise RuntimeError("Invalid exit KD multiplier.")
                        self.interface.send_low_command(q, np.zeros(29), np.zeros(29), q,
                            kp_override=np.zeros(29), kd_override=kd / level)
                        logger.warning("退出 armed：已请求 KP=0、有效 KD={}；需要现场支撑与硬件急停。", kd.tolist())
                    except Exception:
                        logger.exception("无法发送退出阻尼命令，立即使用硬件急停。")
                    finally:
                        self.interface.command_guard = guard
            finally:
                if interface is not None:
                    interface.write_enabled = False
                    interface.command_closed = True
                self._closed = True
                self.use_policy_action = False
                self.motion_clip_progressing = False
                try:
                    if hasattr(self, "_record_file"):
                        self._stop_recording()
                except Exception as exc:
                    logger.error("记录关闭失败：{}", exc)
                if not self._tracking_closed:
                    try:
                        self.tracking.close()
                        self._tracking_closed = True
                    except Exception as exc:
                        logger.error("动捕清理失败，可再次 close：{}", exc)
                executor = getattr(self, "_wbt_executor", None)
                if executor is not None:
                    try:
                        executor.shutdown()
                        self._wbt_executor = None
                    except Exception as exc:
                        logger.error("WBT executor 清理失败：{}", exc)
                for name in ("object_state_sub", "torso_link_pos_sub", "basketball_pose_sub", "waist_pose_sub", "pelvis_pose_sub", "_joint_torque_sub"):
                    node = getattr(self, name, None)
                    if node is not None:
                        try:
                            node.destroy_node()
                            setattr(self, name, None)
                        except Exception as exc:
                            logger.error("{} 清理失败：{}", name, exc)
                clock = getattr(self, "clock_sub", None)
                if clock is not None:
                    try:
                        clock.close()
                        self.clock_sub = None
                    except Exception as exc:
                        logger.error("时钟资源清理失败：{}", exc)
                if self._terminal_settings is not None:
                    try:
                        termios.tcsetattr(self._keyboard_fd, termios.TCSANOW, self._terminal_settings)
                        self._terminal_settings = None
                        self._keyboard_fd = None
                    except Exception as exc:
                        logger.error("终端恢复失败：{}", exc)
                else:
                    self._keyboard_fd = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ReadOnly:
    def __init__(self, config, options):
        self.config, self.options = config, options
        self.interface = RobotInterface(config.robot, config.task.domain_id, config.task.interface,
            False, configure_control_mode=False, write_enabled=False)
        self.interface.await_state()
        self.tracking = Tracking(options)
        urdf = PACKAGE_ROOT.parents[1] / "holosoma/holosoma/data/robots/g1/g1_29dof.urdf"
        self.robot = PinocchioRobot(config.robot, urdf.read_text())
        self.gate = PlacementGate(options, reference_ball_position(options.reference))
        with np.load(options.reference, allow_pickle=False) as data:
            names = data["joint_names"].tolist()
            self.startup_q = data["joint_pos"][0, 7:][[names.index(name) for name in config.robot.dof_names]]
        self.file = None
        if options.record:
            folder = PACKAGE_ROOT.parents[2] / "logs/record"
            folder.mkdir(parents=True, exist_ok=True)
            self.file = (folder / f"tri_readonly_{datetime.now():%Y%m%d_%H%M%S_%f}.txt").open("x")

    def step(self):
        started = time.monotonic()
        state = self.interface.get_low_state()
        validate_state(state, time.monotonic() - started, self.options)
        self.interface.require_fresh_state(self.options.sensor_timeout)
        self.tracking.refresh()
        actual, error, torso = None, None, None
        try:
            self.tracking.require(ball=True)
            pos, quat = self.tracking.torso_pose(state, self.robot)
            actual = matrix_from_quat(quat)[0].T @ (self.tracking.ball.sample["position"] - pos[0])
            torso = {"position": pos, "quat_wxyz": quat}
        except RuntimeError as exc:
            error = str(exc)
        pose_ok = (np.max(np.abs(state[0, 7:36] - self.startup_q)) <= self.options.startup_pose_error
                   and np.max(np.abs(state[0, 42:71])) <= self.options.startup_speed)
        self.gate.check(actual, pose_ok, error)
        snapshot = {"type": "read-only", "motor_mapping": list(enumerate(self.config.robot.dof_names)),
                    "joint2motor": self.config.robot.joint2motor, "robot_state": state, "torso": torso,
                    "mocap_unit": self.options.mocap_unit, "lowstate_tick": self.interface.last_tick,
                    "motor_tau_est": self.interface.motor_tau_est,
                    "pelvis": self.tracking.pelvis.sample, "ball": self.tracking.ball.sample,
                    "placement": self.gate.snapshot, "commands_enabled": False}
        if self.file:
            self.file.write(json.dumps(snapshot, default=ThreePolicy._json_value) + "\n")
        return snapshot

    def run(self):
        logger.info("READ-ONLY：不加载 ONNX，不设置 PR，不发送 lowcmd；Enter 不移动机器人。")
        rate = RateLimiter(self.config.task.rl_rate)
        try:
            iteration = 0
            while True:
                self.step()
                iteration += 1
                if self.options.max_iterations and iteration >= self.options.max_iterations:
                    break
                rate.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        self.interface.command_closed = True
        try:
            if self.file:
                self.file.close()
                self.file = None
        finally:
            self.tracking.close()


def validate_config(config, options):
    if options.mode not in {"read-only", "dry-run", "armed"}:
        raise ValueError("Invalid deployment mode.")
    if config.task.deployment_mode != options.mode:
        raise ValueError("Deployment configuration mode does not match the selected runtime mode.")
    if config.task.interface in {None, "", "lo"} or config.task.use_sim_time:
        raise ValueError("Real deployment needs a physical network interface and no simulation time.")
    if config.task.domain_id != 0:
        raise ValueError("The installed Unitree binding uses DDS domain 0; nonzero task.domain_id is unsupported.")
    if config.robot.robot != "g1" or config.robot.num_joints != 29 or config.robot.sdk_type != "unitree" or config.task.rl_rate != 50:
        raise ValueError("Requires Unitree G1-29DoF at 50 Hz.")
    if config.task.use_joystick:
        raise ValueError("Use keyboard input with --task.no-use-joystick.")
    if not config.robot.motion or config.robot.motion.get("body_name_ref") != ["torso_link"]:
        raise ValueError("This deployment requires torso_link as the reference frame.")
    j2m, m2j = config.robot.joint2motor, config.robot.motor2joint
    if len(j2m) != 29 or len(m2j) != 29 or sorted(j2m) != list(range(29)) or any(m2j[m] != j for j, m in enumerate(j2m)):
        raise ValueError("Joint/motor mappings must be inverse 29-element permutations.")
    positive = [options.tolerance, options.sensor_timeout, options.max_stamp_skew, options.startup_pose_error,
                options.startup_speed, options.ramp_seconds, options.max_command_step, options.max_target_error,
                options.max_joint_speed, options.max_read_seconds]
    positive.append(options.exit_damping_kd)
    if not np.all(np.isfinite(positive)) or min(positive) <= 0 or options.ramp_seconds < 5:
        raise ValueError("Safety limits must be finite/positive and ramp duration at least 5 seconds.")
    if not np.isfinite(options.stable_seconds) or options.stable_seconds < 0 or options.max_iterations < 0:
        raise ValueError("Invalid stability duration or iteration limit.")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    defaults = Options()
    for name, value in vars(defaults).items():
        flag = "--deploy-" + name.replace("_", "-")
        if name == "record":
            parser.add_argument(flag, dest=name, action="store_true")
        else:
            kwargs = {"choices": ("read-only", "dry-run", "armed")} if name == "mode" else {}
            if name == "mocap_unit":
                kwargs["choices"] = ("mm", "cm", "m")
            parser.add_argument(flag, dest=name, type=type(value), default=value, **kwargs)
    parsed, remaining = parser.parse_known_args()
    if "--help" in remaining:
        parser.print_help()
    options = Options(**vars(parsed))
    if options.mode == "read-only" and not any(x.startswith("--task.model-path") for x in remaining):
        remaining += ["--task.model-path", "__read_only__"]
    sys.argv = [sys.argv[0], *remaining]
    config = tyro.cli(AnnotatedInferenceConfig, config=TYRO_CONFIG)
    config = replace(config, task=replace(config.task, deployment_mode=options.mode,
        deployment_write_enabled=False, deployment_configure_control_mode=False))
    validate_config(config, options)
    if options.mode == "armed":
        if not sys.stdin.isatty() or input("Type I_UNDERSTAND_THIS_COMMANDS_A_REAL_ROBOT: ").strip() != "I_UNDERSTAND_THIS_COMMANDS_A_REAL_ROBOT":
            raise RuntimeError("Armed deployment requires explicit acknowledgement in an interactive terminal.")
    runner = ReadOnly(config, options) if options.mode == "read-only" else TriDeployment(config, options)
    try:
        runner.run()
    finally:
        try:
            runner.close()
        finally:
            restore_terminal_settings()


if __name__ == "__main__":
    main()
