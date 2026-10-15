#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(PACKAGE_ROOT.parent), str(PACKAGE_ROOT.parents[1] / "holosoma")]

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from loguru import logger
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from holosoma_inference.config.config_values.robot import g1_29dof
from holosoma_inference.deploy_tri_policy import TrackedPose, torso_from_pelvis
from holosoma_inference.policies.wbt_two_policy import PinocchioRobot
from holosoma_inference.utils.math.quat import matrix_from_quat, xyzw_to_wxyz


STAGES = {1: "static_at_feet", 2: "pelvis_z_up_down", 3: "pelvis_x_front_back", 4: "pelvis_y_left_right"}


@dataclass(frozen=True)
class ProbeOptions:
    interface: str = "eth0"
    pelvis_topic: str = "/pelvis/pose"
    ball_topic: str = "/basketball/pose"
    mocap_unit: str = "mm"
    urdf: str = str(PACKAGE_ROOT.parents[1] / "holosoma/holosoma/data/robots/g1/g1_29dof.urdf")
    output_dir: str = str(PACKAGE_ROOT.parents[2] / "logs/record")
    rate: float = 100.0
    timeout: float = 0.25
    max_pose_skew: float = 0.04
    max_joint_skew: float = 0.05
    queue_depth: int = 100
    max_seconds: float = 0.0
    record_on_start: bool = False


class ObservationInterface:
    def __init__(self, robot_config, domain, interface, use_joystick=False,
                 configure_control_mode=False, write_enabled=False, on_sample=None, depth=100):
        if use_joystick or configure_control_mode or write_enabled:
            raise ValueError("Diagnostic interface must be subscription-only.")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        self.robot_config = robot_config
        self.lock = threading.RLock()
        self.latest = None
        self.last_tick = None
        self.last_tick_at = 0.0
        self.advances = self.sequence = 0
        self.on_sample = on_sample
        self.write_enabled = False
        self.command_closed = False
        self.reader = None
        ChannelFactoryInitialize(domain, interface)
        self.reader = ChannelSubscriber("rt/lowstate", LowState_)
        self.reader.Init(self._callback, depth)

    def _callback(self, msg):
        received = time.monotonic()
        errors, state = [], None
        with self.lock:
            self.sequence += 1
            try:
                tick = int(msg.tick)
                if tick != self.last_tick:
                    forward = self.last_tick is not None and (tick > self.last_tick or (self.last_tick > 2**32 - 1024 and tick < 1024))
                    self.advances = self.advances + 1 if forward else 0
                    self.last_tick, self.last_tick_at = tick, received
                mapping = self.robot_config.joint2motor
                q = np.array([msg.motor_state[i].q for i in mapping])
                dq = np.array([msg.motor_state[i].dq for i in mapping])
                state = np.concatenate([np.zeros(3), np.asarray(msg.imu_state.quaternion), q,
                                        np.zeros(3), np.asarray(msg.imu_state.gyroscope), dq]).reshape(1, -1)
                if state.shape != (1, 71) or not np.all(np.isfinite(state)):
                    errors.append("invalid SDK state shape/NaN")
                elif not 0.8 <= np.linalg.norm(state[0, 3:7]) <= 1.2:
                    errors.append("invalid SDK IMU quaternion")
                if tick <= 0 or self.advances < 1:
                    errors.append("SDK tick 尚未推进或已重置")
            except Exception as exc:
                errors.append(str(exc))
            sample = {"type": "sdk", "sequence": self.sequence, "received_at": received,
                      "observed_update_at": self.last_tick_at, "tick": self.last_tick,
                      "state": state, "valid": not errors, "reasons": errors}
            self.latest = sample
        if self.on_sample is not None:
            self.on_sample(sample)

    def get_low_state(self):
        with self.lock:
            return None if self.latest is None or self.latest["state"] is None else self.latest["state"].copy()

    def require_fresh_state(self, timeout):
        with self.lock:
            if self.latest is None or not self.latest["valid"] or time.monotonic() - self.last_tick_at > timeout:
                raise RuntimeError("SDK lowstate missing/invalid or tick stalled")

    def close(self):
        self.command_closed = True
        if self.reader is not None:
            self.reader.Close()
            self.reader = None

    def send_low_command(self, *args, **kwargs):
        raise RuntimeError("This diagnostic never sends motor commands.")

    def _send_binding_command(self, *args, **kwargs):
        raise RuntimeError("Native motor command writes are prohibited in this diagnostic.")


def clean_json(value):
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


class Rates:
    def __init__(self):
        self.count = 0
        self.first = self.last = None
        self.last_stamp = None
        self.receive_dt_sum = self.receive_dt_max = 0.0
        self.source_dt_sum = 0.0
        self.source_intervals = self.duplicates = self.backwards = self.large_gaps = 0
        self.source_gaps = 0

    def observe(self, received, stamp=None):
        if self.first is None:
            self.first = received
        if self.last is not None:
            dt = received - self.last
            self.receive_dt_sum += dt
            self.receive_dt_max = max(self.receive_dt_max, dt)
            self.large_gaps += dt > 0.1
        self.last = received
        self.count += 1
        if stamp is not None and stamp > 0:
            if self.last_stamp is not None:
                dt = stamp - self.last_stamp
                if dt == 0:
                    self.duplicates += 1
                elif dt < 0:
                    self.backwards += 1
                else:
                    self.source_dt_sum += dt
                    self.source_intervals += 1
                    self.source_gaps += dt > 0.1
            self.last_stamp = stamp

    def summary(self, now, live=True):
        intervals = self.count - 1
        receive_hz = intervals / self.receive_dt_sum if self.receive_dt_sum > 0 else None
        if live and self.last is not None and now - self.last > 1:
            receive_hz = 0.0
        return {"count": self.count, "receive_hz": receive_hz,
                "first_received_at": self.first, "last_received_at": self.last,
                "source_hz": self.source_intervals / self.source_dt_sum if self.source_dt_sum > 0 else None,
                "mean_receive_dt": self.receive_dt_sum / intervals if intervals > 0 else None,
                "max_receive_dt": self.receive_dt_max, "duplicate_stamps": self.duplicates,
                "backward_stamps": self.backwards, "receive_gaps_over_100ms": self.large_gaps,
                "source_gaps_over_100ms": self.source_gaps,
                "last_receive_age": now - self.last if self.last is not None else None}


class Moments:
    def __init__(self):
        self.n = 0
        self.mean = np.zeros(3)
        self.m2 = np.zeros(3)
        self.minimum = np.full(3, np.inf)
        self.maximum = np.full(3, -np.inf)

    def add(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.minimum = np.minimum(self.minimum, value)
        self.maximum = np.maximum(self.maximum, value)

    def summary(self):
        if not self.n:
            return {"count": 0}
        return {"count": self.n, "mean_m": self.mean, "std_m": np.sqrt(np.maximum(0, self.m2 / max(1, self.n - 1))),
                "min_m": self.minimum, "max_m": self.maximum, "range_m": self.maximum - self.minimum}


def relative_positions(state, robot, pelvis, ball):
    torso_pos, torso_quat = torso_from_pelvis(state, robot, pelvis)
    pelvis_rot = matrix_from_quat(xyzw_to_wxyz(pelvis["quat_xyzw"].reshape(1, 4)))[0]
    torso_rot = matrix_from_quat(torso_quat)[0]
    return {"ball_pos_pelvis": pelvis_rot.T @ (ball["position"] - pelvis["position"]),
            "obj_pos_b": torso_rot.T @ (ball["position"] - torso_pos[0]),
            "torso_pos_w": torso_pos[0], "torso_quat_wxyz": torso_quat[0]}


class DiagnosticPose(TrackedPose):
    def __init__(self, owner, stream, topic):
        Node.__init__(self, f"obj_pos_b_probe_{stream}_{os.getpid()}_{id(owner):x}")
        self.owner, self.stream = owner, stream
        self.scale = {"mm": 0.001, "cm": 0.01, "m": 1.0}[owner.options.mocap_unit]
        self.sample = None
        self.last_invalid = None
        self.orientation_required = stream == "pelvis"
        self.sequence = 0
        qos = QoSProfile(depth=owner.options.queue_depth, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PoseStamped, topic, self._callback, qos)

    def _callback(self, msg):
        received = time.monotonic()
        with self.owner.lock:
            admitted = received >= self.owner.admission_after
            self.sequence += 1
            previous = self.sample
            super()._callback(msg)
            accepted = self.sample is not None and self.sample is not previous
            stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            self.owner.observe_rate(self.stream, received, stamp_ns * 1e-9, segment=admitted)
            if accepted and admitted:
                sample = dict(self.sample, sequence=self.sequence, received_at=received, stamp_ns=stamp_ns)
                self.owner.poses[self.stream].append(sample)
            p, q = msg.pose.position, msg.pose.orientation
            self.owner.write({"type": "pose", "stream": self.stream, "sequence": self.sequence,
                "received_at": received, "stamp_ns": stamp_ns, "frame_id": msg.header.frame_id,
                "raw_position": [p.x, p.y, p.z], "position_m": np.array([p.x, p.y, p.z]) * self.scale,
                "raw_quat_xyzw": [q.x, q.y, q.z, q.w], "accepted": accepted, "admitted_to_phase": admitted, "reason": self.last_invalid})


class ObjPosProbe:
    def __init__(self, options):
        if options.interface in {"", "lo"}:
            raise ValueError("Specify the physical G1 network interface.")
        if options.mocap_unit not in {"mm", "cm", "m"}:
            raise ValueError("Invalid mocap unit.")
        if not np.all(np.isfinite([options.rate, options.timeout, options.max_pose_skew, options.max_joint_skew, options.max_seconds])):
            raise ValueError("Timing parameters must be finite.")
        if min(options.rate, options.timeout, options.max_pose_skew, options.max_joint_skew) <= 0 or options.max_seconds < 0 or options.queue_depth < 1:
            raise ValueError("Invalid timing/depth configuration.")
        self.options = options
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.executor = None
        self.pelvis = self.ball = self.interface = None
        self.file = self.path = None
        self.stage = 1
        self.segment_id = 0
        self.segments = []
        self.current_segment = None
        self.rates = {k: Rates() for k in ("pelvis", "ball", "sdk_receive", "sdk_poll", "sdk_observed_updates", "valid_computations")}
        self.poses = {"pelvis": deque(maxlen=options.queue_depth), "ball": deque(maxlen=options.queue_depth)}
        self.states = deque(maxlen=options.queue_depth)
        self.sdk_sequence = 0
        self.last_sdk_poll = None
        self.admission_after = -float("inf")
        self.last_pair = None
        self.last_result = {"valid": False, "reasons": ["等待数据"]}
        self.background_error = None
        self.fd = self.terminal = None
        self.closed = False
        try:
            text = Path(options.urdf).read_text()
            self.urdf_sha256 = hashlib.sha256(text.encode()).hexdigest()
            self.robot = PinocchioRobot(g1_29dof, text)
            self.interface = ObservationInterface(g1_29dof, 0, options.interface, False,
                configure_control_mode=False, write_enabled=False, on_sample=self.capture_sdk, depth=options.queue_depth)
            if not rclpy.ok():
                rclpy.init(args=None)
            self.pelvis = DiagnosticPose(self, "pelvis", options.pelvis_topic)
            self.ball = DiagnosticPose(self, "ball", options.ball_topic)
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.pelvis)
            self.executor.add_node(self.ball)
        except Exception:
            self.close()
            raise

    def observe_rate(self, name, received, stamp=None, segment=True):
        self.rates[name].observe(received, stamp)
        if segment and self.current_segment is not None:
            self.current_segment["rates"][name].observe(received, stamp)

    def write(self, row):
        if self.file is not None:
            self.file.write(json.dumps(clean_json(dict(row, stage=self.stage, segment_id=self.segment_id)), allow_nan=False) + "\n")

    def new_segment(self):
        self.segment_id += 1
        self.admission_after = time.monotonic()
        self.current_segment = {"id": self.segment_id, "stage": self.stage, "label": STAGES[self.stage],
            "started_at": self.admission_after,
            "rates": {k: Rates() for k in self.rates}, "pelvis": Moments(), "torso": Moments(), "invalid": 0}
        self.segments.append(self.current_segment)

    def toggle_recording(self):
        with self.lock:
            if self.file is not None:
                self.finish_recording()
                return
            folder = Path(self.options.output_dir).resolve()
            if not folder.is_relative_to(PACKAGE_ROOT.parents[2]):
                raise ValueError("Recordings must remain under /home/sustech/holosoma.")
            folder.mkdir(parents=True, exist_ok=True)
            self.path = folder / f"obj_pos_b_test_{datetime.now():%Y%m%d_%H%M%S_%f}.txt"
            self.file = self.path.open("x", encoding="utf-8")
            self.segments = []
            self.new_segment()
            for buffer in self.poses.values():
                buffer.clear()
            self.states.clear()
            self.write({"type": "metadata", "options": asdict(self.options), "urdf_sha256": self.urdf_sha256,
                "joint_names": g1_29dof.dof_names, "joint2motor": g1_29dof.joint2motor, "stages": STAGES,
                "joint_state_source": "Unitree DDS measured q", "controls_prohibited": True,
                "time_basis": "receive times monotonic; header source stamps ROS clock; SDK updates observed, not DDS publish Hz"})
            self.file.flush()
            self.last_pair = None
            logger.info("开始记录：{}", self.path)

    def segment_summaries(self):
        now = time.monotonic()
        return [{"id": s["id"], "stage": s["stage"], "label": s["label"], "started_at": s["started_at"], "invalid": s["invalid"],
                 "ball_pos_pelvis": s["pelvis"].summary(), "obj_pos_b": s["torso"].summary(),
                 "rates": {k: v.summary(now, live=False) for k, v in s["rates"].items()}} for s in self.segments]

    def finish_recording(self):
        if self.file is None:
            return
        try:
            summaries = self.segment_summaries()
            self.write({"type": "summary", "segments": summaries})
            self.file.flush()
            for s in summaries:
                logger.info("阶段 {}：有效={}，pelvis 各轴范围(m)={}，torso 各轴范围(m)={}",
                    s["label"], s["obj_pos_b"]["count"], s["ball_pos_pelvis"].get("range_m"), s["obj_pos_b"].get("range_m"))
        finally:
            self.file.close()
            self.file = None
            self.current_segment = None
        logger.info("记录已停止：{}", self.path)

    def handle_key(self, key):
        with self.lock:
            if key == "r":
                self.toggle_recording()
            elif key in {"1", "2", "3", "4"}:
                self.stage = int(key)
                self.admission_after = time.monotonic()
                if self.file is not None:
                    self.new_segment()
                    self.write({"type": "stage_event", "received_at": time.monotonic(), "label": STAGES[self.stage]})
                for buffer in self.poses.values():
                    buffer.clear()
                self.states.clear()
                self.last_pair = None
                logger.info("当前测试：{}", STAGES[self.stage])
            elif key in {"q", "\x1b", "\x04"}:
                self.stop.set()

    def capture_sdk(self, sample):
        with self.lock:
            if not self.closed:
                admitted = sample["received_at"] >= self.admission_after and sample["observed_update_at"] >= self.admission_after
                if admitted:
                    self.observe_rate("sdk_receive", sample["received_at"])
                    self.states.append(sample)
                else:
                    self.rates["sdk_receive"].observe(sample["received_at"])
                self.write(dict(sample, admitted_to_phase=admitted))

    def poll_sdk(self):
        started = time.monotonic()
        previous = getattr(self, "_last_polled_tick", None)
        errors, state = [], None
        try:
            state = self.interface.get_low_state()
            self.interface.require_fresh_state(self.options.timeout)
            if state.shape != (1, 71) or not np.all(np.isfinite(state)):
                errors.append("invalid SDK state shape/NaN")
            elif not 0.8 <= np.linalg.norm(state[0, 3:7]) <= 1.2:
                errors.append("invalid SDK IMU quaternion")
        except Exception as exc:
            errors.append(str(exc))
        received = time.monotonic()
        self.observe_rate("sdk_poll", received)
        tick = getattr(self.interface, "last_tick", None)
        if previous is not None and tick != previous:
            self.observe_rate("sdk_observed_updates", received)
        self._last_polled_tick = tick
        self.sdk_sequence += 1
        sample = {"type": "sdk_poll", "sequence": self.sdk_sequence, "received_at": received, "tick": tick,
                  "read_seconds": received - started, "state": state, "valid": not errors, "reasons": errors}
        self.last_sdk_poll = sample
        self.write(sample)
        return sample

    def compute(self):
        now = time.monotonic()
        reasons = []
        if self.pelvis.sample is None or self.ball.sample is None or not self.poses["ball"] or not self.poses["pelvis"] or not self.states:
            return {"type": "calculation", "valid": False, "reasons": ["等待有效 pelvis/basketball/SDK 数据"],
                    "received_at": now, "obj_pos_b": None, "ball_pos_pelvis": None}
        ball = self.poses["ball"][-1]
        pelvis = min(self.poses["pelvis"], key=lambda p: abs(p["stamp_ns"] - ball["stamp_ns"]))
        midpoint = (pelvis["received_at"] + ball["received_at"]) / 2
        sdk = min(self.states, key=lambda s: abs(s["observed_update_at"] - midpoint))
        key = (pelvis["sequence"], ball["sequence"])
        source_now = self.pelvis.get_clock().now().nanoseconds * 1e-9
        for label, sample in (("pelvis", pelvis), ("ball", ball)):
            if now - sample["received_at"] > self.options.timeout:
                reasons.append(label + " 接收过期")
            if sample["stamp"] <= 0 or not -self.options.max_pose_skew <= source_now - sample["stamp"] <= self.options.timeout:
                reasons.append(label + " 源时间戳过期/本机时钟不一致")
        if not pelvis["frame_id"] or pelvis["frame_id"] != ball["frame_id"]:
            reasons.append("global frame_id 不一致或为空")
        skew = abs(pelvis["stamp_ns"] - ball["stamp_ns"]) * 1e-9
        joint_skew = abs(sdk["observed_update_at"] - midpoint)
        if skew > self.options.max_pose_skew:
            reasons.append("动捕时间差过大")
        if (joint_skew > self.options.max_joint_skew or now - sdk["observed_update_at"] > self.options.timeout
                or not sdk["valid"] or not self.states[-1]["valid"] or (self.last_sdk_poll is not None and not self.last_sdk_poll["valid"])):
            reasons.append("SDK 配对无效或接收时间差过大")
        if key == self.last_pair and not reasons and self.last_result["valid"]:
            return None
        row = {"type": "calculation", "received_at": now, "valid": not reasons, "reasons": reasons,
               "pelvis_sequence": pelvis["sequence"], "ball_sequence": ball["sequence"], "sdk_sequence": sdk["sequence"],
               "pose_stamp_skew": skew, "joint_receive_skew": joint_skew,
               "obj_pos_b": None, "ball_pos_pelvis": None}
        if not reasons:
            row.update(relative_positions(sdk["state"], self.robot, pelvis, ball))
            row["waist_yaw_roll_pitch"] = sdk["state"][0, 19:22].copy()
            if not all(np.all(np.isfinite(row[k])) for k in ("obj_pos_b", "ball_pos_pelvis", "torso_pos_w", "torso_quat_wxyz")):
                row.update(valid=False, obj_pos_b=None, ball_pos_pelvis=None, reasons=["FK output non-finite"])
                return row
            if key != self.last_pair:
                self.observe_rate("valid_computations", now)
            if self.current_segment is not None and key != self.last_pair:
                self.current_segment["pelvis"].add(row["ball_pos_pelvis"])
                self.current_segment["torso"].add(row["obj_pos_b"])
            self.last_pair = key
        return row

    def step(self):
        with self.lock:
            self.poll_sdk()
            result = self.compute()
            if result is not None:
                self.last_result = result
                if not result["valid"] and self.current_segment is not None:
                    self.current_segment["invalid"] += 1
                self.write(result)
            return result

    def ros_worker(self):
        try:
            while not self.stop.is_set() and rclpy.ok():
                self.executor.spin_once(timeout_sec=0.02)
        except Exception as exc:
            if not self.stop.is_set() and rclpy.ok():
                with self.lock:
                    self.background_error = str(exc)
                self.stop.set()

    def run(self):
        logger.info("仅观测：无 ONNX、无 PR、无 lowcmd。r 录制；1 静止/2 Z/3 X/4 Y；q 退出。")
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.terminal = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        self.thread = threading.Thread(target=self.ros_worker, daemon=True)
        self.thread.start()
        started, last_report = time.monotonic(), -float("inf")
        try:
            if self.options.record_on_start:
                self.toggle_recording()
            while not self.stop.is_set() and rclpy.ok():
                cycle = time.monotonic()
                if self.fd is not None:
                    for _ in range(16):
                        if not select.select([self.fd], [], [], 0)[0]:
                            break
                        value = os.read(self.fd, 1)
                        self.handle_key(value.decode(errors="ignore") if value else "q")
                self.step()
                with self.lock:
                    if cycle - last_report >= 1:
                        report_time = time.monotonic()
                        stats = {k: v.summary(report_time) for k, v in self.rates.items()}
                        logger.info("{} valid={} pelvis_b={} torso_b={} 原因={} | 累计平均Hz pelvis接收/源={}/{}，ball接收/源={}/{}，SDK接收/采样观测更新={}/{}",
                            STAGES[self.stage], self.last_result["valid"], self.last_result.get("ball_pos_pelvis"),
                            self.last_result.get("obj_pos_b"), self.last_result["reasons"], stats["pelvis"]["receive_hz"],
                            stats["pelvis"]["source_hz"], stats["ball"]["receive_hz"], stats["ball"]["source_hz"],
                            stats["sdk_receive"]["receive_hz"], stats["sdk_observed_updates"]["receive_hz"])
                        if self.file is not None:
                            self.write({"type": "rates", "scope": "process_lifetime", "received_at": report_time, "streams": stats,
                                        "segment_streams": {k: v.summary(report_time, live=False) for k, v in self.current_segment["rates"].items()}})
                            self.file.flush()
                        last_report = cycle
                if self.options.max_seconds and cycle - started >= self.options.max_seconds:
                    break
                time.sleep(max(0, 1 / self.options.rate - (time.monotonic() - cycle)))
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
        if self.background_error:
            raise RuntimeError(self.background_error)

    def close(self):
        if self.closed:
            return
        self.stop.set()
        errors = []
        if self.terminal is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSANOW, self.terminal)
                self.terminal = None
            except Exception as exc:
                errors.append(exc)
        if self.interface is not None:
            self.interface.write_enabled = False
            self.interface.command_closed = True
            try:
                self.interface.close()
            except Exception as exc:
                errors.append(exc)
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("ROS recording thread did not stop; resources retained.")
        try:
            with self.lock:
                self.finish_recording()
        except Exception as exc:
            errors.append(exc)
        finally:
            if self.interface is not None:
                self.interface.write_enabled = False
                self.interface.command_closed = True
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
            self.closed = not errors
        if errors:
            raise RuntimeError(f"Diagnostic cleanup incomplete: {errors}")


def main():
    parser = argparse.ArgumentParser(description="ROS mocap + read-only Unitree SDK obj_pos_b diagnostic")
    for name, value in asdict(ProbeOptions()).items():
        if isinstance(value, bool):
            parser.add_argument("--" + name.replace("_", "-"), action="store_true")
        else:
            parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    options = ProbeOptions(**vars(parser.parse_args()))
    probe = ObjPosProbe(options)
    try:
        probe.run()
    finally:
        probe.close()


if __name__ == "__main__":
    main()
