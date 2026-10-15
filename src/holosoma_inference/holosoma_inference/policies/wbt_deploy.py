import json
import sys
import time
import threading
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pinocchio as pin
from defusedxml import ElementTree
from loguru import logger
from termcolor import colored

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some pure-inference environments
    torch = None

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.policies import BasePolicy
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.math.misc import get_index_of_a_in_b
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_mul,
    quat_rotate_inverse,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


class PinocchioRobot:
    def __init__(self, robot_cfg: RobotConfig, urdf_text: str):
        # create pinocchio robot
        xml_text = self._create_xml_from_urdf(urdf_text)
        self.robot_model = pin.buildModelFromXML(xml_text, pin.JointModelFreeFlyer())
        self.robot_data = self.robot_model.createData()

        # get joint names in pinocchio robot and real robot
        joint_names_in_real_robot = robot_cfg.dof_names
        joint_names_in_pinocchio_robot = [
            name for name in self.robot_model.names if name not in ["universe", "root_joint"]
        ]
        assert len(joint_names_in_pinocchio_robot) == len(joint_names_in_real_robot), (
            "The number of joints in the pinocchio robot and the real robot are not the same"
        )
        self.real2pinocchio_index = get_index_of_a_in_b(joint_names_in_pinocchio_robot, joint_names_in_real_robot)

        # get ref body frame id in pinocchio robot
        self.ref_body_frame_id = self.robot_model.getFrameId(robot_cfg.motion["body_name_ref"][0])

    def fk_and_get_ref_body_pose_in_world(self, configuration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回参考刚体在世界坐标系下的位置和姿态。

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (pos_w, quat_xyzw)，shape 分别为 (1, 3) 和 (1, 4)。
        """
        # forward kinematics
        pin.framesForwardKinematics(self.robot_model, self.robot_data, configuration)

        # get ref body pose in world
        ref_body_pose_in_world = self.robot_data.oMf[self.ref_body_frame_id]
        position = np.expand_dims(np.asarray(ref_body_pose_in_world.translation), axis=0)  # (1, 3)
        quaternion = pin.Quaternion(ref_body_pose_in_world.rotation)  # (4, )

        return position, np.expand_dims(quaternion.coeffs(), axis=0)  # xyzw, (1, 4)

    def fk_and_get_ref_body_orientation_in_world(self, configuration: np.ndarray) -> np.ndarray:
        _, quat_xyzw = self.fk_and_get_ref_body_pose_in_world(configuration)
        return quat_xyzw

    @staticmethod
    def _create_xml_from_urdf(urdf_text: str) -> str:
        """Strip visuals/collisions from URDF text and return XML text."""
        root = ElementTree.fromstring(urdf_text)

        def _is_visual_or_collision(tag: str) -> bool:
            # Handle optional XML namespaces by only checking the suffix after '}'.
            return tag.split("}")[-1] in {"visual", "collision"}

        for parent in root.iter():
            for child in list(parent):
                if _is_visual_or_collision(child.tag):
                    parent.remove(child)

        xml_text = ElementTree.tostring(root, encoding="unicode")
        if not xml_text.lstrip().startswith("<?xml"):
            xml_text = '<?xml version="1.0"?>\n' + xml_text
        return xml_text


class ObjectStateSub(Node):
    def __init__(self, topic_name: str = "/object_state"):
        super().__init__("wbt_object_state_subscriber")
        self.latest_pos_w = np.zeros((1, 3), dtype=np.float32)
        self.latest_ori_w = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        self.latest_lin_vel_w = np.zeros((1, 3), dtype=np.float32)
        self.received = False
        self.create_subscription(Float32MultiArray, topic_name, self._callback, 10)

    def _callback(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size < 12:
            self.get_logger().warning(f"/object_state data size is {data.size}, expected at least 12")
            return
        self.latest_pos_w = data[0:3].reshape(1, 3)
        self.latest_ori_w = data[3:9].reshape(1, 6)
        self.latest_lin_vel_w = data[9:12].reshape(1, 3)
        self.received = True


class TorsoLinkPosSub(Node):
    def __init__(self, topic_name: str = "/torso_link_pos"):
        super().__init__("wbt_torso_link_pos_subscriber")
        self.latest_pos_w = np.zeros((1, 3), dtype=np.float32)
        self.received = False
        self.create_subscription(Float32MultiArray, topic_name, self._callback, 10)

    def _callback(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size < 3:
            self.get_logger().warning(f"/torso_link_pos data size is {data.size}, expected at least 3")
            return
        self.latest_pos_w = data[0:3].reshape(1, 3)
        self.received = True


class MocapPoseSub(Node):
    """Subscribe to a geometry_msgs/PoseStamped mocap rigid-body pose.

    中文说明：
    - 动捕系统发布的 PoseStamped 里，position 通常是世界系位置；
    - orientation 在 ROS 中是 xyzw 顺序；
    - position_scale=0.001 时会把 mm 自动转成 m；
    - 安全掉帧判定改为以“本机接收时间间隔 receive_dt”为主，
      header.stamp 只作为调试参考，避免动捕时间戳抖动导致误判每秒大量掉帧。
    """

    def __init__(
        self,
        topic_name: str,
        node_name: str,
        position_scale: float = 1.0,
        expected_hz: float = 120.0,
        drop_gap_sec: float | None = None,
    ):
        super().__init__(node_name)
        self.topic_name = topic_name
        self.position_scale = float(position_scale)
        self.expected_hz = float(expected_hz)
        self.expected_dt_sec = 1.0 / max(self.expected_hz, 1e-6)
        # 默认 3 帧以上没有收到新回调才认为发生真实接收掉帧。
        # 120Hz 下 expected_dt≈8.33ms，所以默认 drop_gap≈25ms。
        self.drop_gap_sec = float(drop_gap_sec) if drop_gap_sec is not None else 3.0 * self.expected_dt_sec

        self.latest_pos_w = np.zeros((1, 3), dtype=np.float32)
        self.latest_quat_xyzw = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.received = False

        self.message_count = 0
        self.invalid_count = 0

        # drop_count / drop_event_count 只记录“接收时间间隔过大”的安全相关掉帧。
        self.drop_count = 0
        self.drop_event_count = 0

        # stamp_* 只作为调试信息，不再直接阻止 policy。
        self.stamp_gap_count = 0
        self.stamp_gap_event_count = 0
        self.stamp_jump_back_count = 0

        self.last_receive_time_monotonic: float | None = None
        self.last_stamp_sec: float | None = None

        # last_dt_sec 兼容旧日志字段，当前表示 receive_dt，而不是 header.stamp dt。
        self.last_dt_sec: float | None = None
        self.last_receive_dt_sec: float | None = None
        self.max_receive_dt_sec: float = 0.0
        self.last_stamp_dt_sec: float | None = None
        self.max_stamp_dt_sec: float = 0.0
        self.last_drop_reason: str = "never_received"
        self.last_stamp_warning: str = "none"

        self.create_subscription(PoseStamped, topic_name, self._callback, 10)

    def _callback(self, msg: PoseStamped) -> None:
        now = time.monotonic()
        pos = msg.pose.position
        quat = msg.pose.orientation

        quat_xyzw = np.array([[quat.x, quat.y, quat.z, quat.w]], dtype=np.float32)
        norm = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
        if not np.all(np.isfinite(quat_xyzw)) or float(norm[0, 0]) < 1e-8:
            self.invalid_count += 1
            self.last_drop_reason = "invalid_quaternion"
            self.get_logger().warning(f"{self.topic_name} quaternion is invalid; keeping previous pose.")
            return

        # 1) 安全相关：用本机接收时间间隔判断是否真的长时间没有收到新消息。
        if self.last_receive_time_monotonic is not None:
            receive_dt = now - self.last_receive_time_monotonic
            self.last_receive_dt_sec = receive_dt
            self.last_dt_sec = receive_dt
            self.max_receive_dt_sec = max(self.max_receive_dt_sec, float(abs(receive_dt)))
            if receive_dt > self.drop_gap_sec:
                estimated_missing = max(1, int(round(receive_dt / self.expected_dt_sec)) - 1)
                self.drop_event_count += 1
                self.drop_count += estimated_missing
                self.last_drop_reason = (
                    f"large_receive_gap={receive_dt:.4f}s > {self.drop_gap_sec:.4f}s, "
                    f"estimated_missing={estimated_missing}"
                )
        else:
            self.last_dt_sec = None
            self.last_receive_dt_sec = None

        # 2) 调试相关：记录 header.stamp 的抖动，但不把它作为安全掉帧依据。
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if stamp_sec > 0.0 and self.last_stamp_sec is not None:
            stamp_dt = stamp_sec - self.last_stamp_sec
            self.last_stamp_dt_sec = stamp_dt
            self.max_stamp_dt_sec = max(self.max_stamp_dt_sec, float(abs(stamp_dt)))
            if stamp_dt < -1e-6:
                self.stamp_jump_back_count += 1
                self.last_stamp_warning = f"timestamp_jump_back={stamp_dt:.4f}s"
            elif stamp_dt > self.drop_gap_sec:
                estimated_missing = max(1, int(round(stamp_dt / self.expected_dt_sec)) - 1)
                self.stamp_gap_event_count += 1
                self.stamp_gap_count += estimated_missing
                self.last_stamp_warning = (
                    f"large_stamp_gap={stamp_dt:.4f}s > {self.drop_gap_sec:.4f}s, "
                    f"estimated_missing={estimated_missing}"
                )
        elif stamp_sec <= 0.0:
            self.last_stamp_warning = "header_stamp_zero_or_invalid"

        self.latest_pos_w = np.array([[pos.x, pos.y, pos.z]], dtype=np.float32) * self.position_scale
        self.latest_quat_xyzw = quat_xyzw / norm
        self.received = True
        self.message_count += 1
        self.last_receive_time_monotonic = now
        if stamp_sec > 0.0:
            self.last_stamp_sec = stamp_sec

    def age_sec(self) -> float:
        if self.last_receive_time_monotonic is None:
            return float("inf")
        return float(time.monotonic() - self.last_receive_time_monotonic)

    def snapshot(self) -> dict[str, float | int | bool | str]:
        return {
            "topic": self.topic_name,
            "received": self.received,
            "age_sec": self.age_sec(),
            "message_count": self.message_count,
            "drop_count": self.drop_count,
            "drop_event_count": self.drop_event_count,
            "invalid_count": self.invalid_count,
            "last_dt_sec": float("nan") if self.last_dt_sec is None else float(self.last_dt_sec),
            "last_receive_dt_sec": float("nan") if self.last_receive_dt_sec is None else float(self.last_receive_dt_sec),
            "max_receive_dt_sec": float(self.max_receive_dt_sec),
            "last_stamp_dt_sec": float("nan") if self.last_stamp_dt_sec is None else float(self.last_stamp_dt_sec),
            "max_stamp_dt_sec": float(self.max_stamp_dt_sec),
            "stamp_gap_count": self.stamp_gap_count,
            "stamp_gap_event_count": self.stamp_gap_event_count,
            "stamp_jump_back_count": self.stamp_jump_back_count,
            "last_drop_reason": self.last_drop_reason,
            "last_stamp_warning": self.last_stamp_warning,
        }


class NoFreshMocapError(RuntimeError):
    """Raised when real-only mode refuses to use sim fallback because mocap is missing/stale."""


class WholeBodyTrackingPolicy(BasePolicy):
    def __init__(self, config: InferenceConfig):
        # initialize timestep
        self.motion_timestep = 0
        self.motion_clip_progressing = False
        self.motion_start_timestep = None
        self.motion_command_t = None
        self.ref_quat_xyzw_t = None
        self.motion_command_0 = None
        self.ref_quat_xyzw_0 = None

        # Calculate timestep interval from rl_rate (e.g., 50Hz = 20ms intervals)
        self.timestep_interval_ms = 1000.0 / config.task.rl_rate

        # Initialize clock subscriber for synchronization
        self.clock_sub = ClockSub()
        self.clock_sub.start()
        self._last_clock_reading: int | None = None

        # Read use_sim_time from config
        self.use_sim_time = config.task.use_sim_time

        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0

        if not rclpy.ok():
            rclpy.init(args=None)

        # 中文说明：
        # 1) 仿真模式保留原来的 /object_state 和 /torso_link_pos 订阅；
        # 2) 实机动捕模式优先使用 /basketball/pose 和 /waist/pose；
        # 3) 你的动捕 position 截图看起来是 mm，所以默认 scale=0.001 转成 m。
        self.object_state_sub = ObjectStateSub(topic_name="/object_state")
        self.torso_link_pos_sub = TorsoLinkPosSub(topic_name="/torso_link_pos")
        self.mocap_position_scale = 0.001
        # 动捕频率：120Hz，理论帧间隔 8.33ms。
        self.mocap_expected_hz = 120.0
        self.mocap_expected_dt_sec = 1.0 / self.mocap_expected_hz
        # 安全判定阈值：
        # - receive_dt 超过 3 帧（约 25ms）才认为发生真实接收掉帧；
        # - 超过 6 帧（约 50ms）没有新消息则认为数据不新鲜。
        # header.stamp 的抖动只做调试显示，不直接阻止 policy。
        self.mocap_drop_gap_sec = 3.0 * self.mocap_expected_dt_sec
        self.mocap_stale_timeout_sec = 6.0 * self.mocap_expected_dt_sec
        self.basketball_pose_sub = MocapPoseSub(
            topic_name="/basketball/pose",
            node_name="wbt_basketball_pose_subscriber",
            position_scale=self.mocap_position_scale,
            expected_hz=self.mocap_expected_hz,
            drop_gap_sec=self.mocap_drop_gap_sec,
        )
        self.waist_pose_sub = MocapPoseSub(
            topic_name="/waist/pose",
            node_name="wbt_waist_pose_subscriber",
            position_scale=self.mocap_position_scale,
            expected_hz=self.mocap_expected_hz,
            drop_gap_sec=self.mocap_drop_gap_sec,
        )

        # torso_link 相对于 waist_yaw_link 的近似平移。
        # 中文说明：原 URDF z 方向约 0.044 m，但由于动捕标不是严格贴在 waist_yaw_link 原点，
        # 这里按你的要求改为 0.034 m；保留 x 方向 3.9635 mm 偏移。
        # 该 offset 表达在 waist_yaw_link / waist 动捕刚体坐标系下。
        self.torso_offset_in_waist = np.array([[-0.0039635, 0.0, 0.034]], dtype=np.float32)
        self._mocap_wait_warned = False

        # Debug: print which data source is used for obj_pos_b.
        self.debug_obj_pos_b_source_enabled = True
        self.debug_obj_pos_b_every_n = 50
        self._debug_obj_pos_b_counter = 0
        self._last_obj_pos_b_source = None

        # 真机安全模式：默认禁止任何 MuJoCo/sim fallback。
        # 只有显式按 m 进入 sim-debug 模式后，才允许使用仿真 fallback。
        self.real_only_mode = True
        self.allow_sim_fallback = False
        self.real_mode_key = "r"
        self.sim_mode_key = "m"
        self.mocap_status_key = "p"

        # Joystick / Unitree remote aliases for the safety/debug keys.
        # 中文说明：这里按 InterfaceWrapper._default_wc_key_map() 的宇树遥控器按键名来绑定。
        # - select+X：等价于键盘 i，只开关动捕状态监测，不会控制机器人；
        # - select+Y：等价于键盘 r，进入 real-only 真机安全模式；
        # - select+B：等价于键盘 m，进入 sim-debug 模式，仅用于 MuJoCo 仿真调试。
        #
        # 保留原来的普通按键语义：A 启动 policy、B 停止、Y 初始化、select 切换 policy、L1+R1 退出。
        # 注意：下面统一用 lower() 后的字符串比较，所以写成小写。
        self.joystick_mocap_status_keys = {"select+x"}
        self.joystick_real_mode_keys = {"select+y"}
        self.joystick_sim_mode_keys = {"select+b"}
        self.debug_joystick_key_enabled = True

        self.mocap_status_monitor_enabled = False
        self.mocap_status_print_period_sec = 1.0
        self._mocap_status_last_print_time = 0.0
        # 中文说明：所有 ROS2 subscriber 统一交给一个私有 SingleThreadedExecutor 处理。
        # 这样既保留 /basketball/pose、/waist/pose、/object_state、/torso_link_pos 的接收，
        # 又避免主控制循环、动捕接收线程、monitor 线程同时调用 rclpy.spin_once(...)
        # 造成 ROS2 默认 executor 的 "generator already executing" 并发错误。
        self._mocap_spin_lock = threading.RLock()
        self._ros_executor = SingleThreadedExecutor()
        self._ros_executor_nodes_added = False
        self._mocap_monitor_thread = None
        self._mocap_monitor_stop = False
        self._mocap_status_prev_counts = {
            "basketball_msg": 0,
            "waist_msg": 0,
            "basketball_drop": 0,
            "waist_drop": 0,
        }
        self.mocap_preflight_duration_sec = 5.0
        self.mocap_preflight_poll_sec = 0.005
        self._mocap_drop_snapshot = {"basketball": 0, "waist": 0}
        self._holding_due_to_mocap_loss = False
        self._last_safe_scaled_policy_action = None

        # 中文说明：动捕 topic 是 120Hz 级别，不能只在 monitor 打印时才 spin。
        # 这里单独启动一个常驻接收线程，高频处理 /basketball/pose 和 /waist/pose 的 ROS2 回调；
        # monitor 只负责每秒统计/打印，不再承担接收消息的职责。
        self.mocap_spin_period_sec = 0.001  # 约 1000Hz 轮询，足够覆盖 120Hz*2 个 PoseStamped topic
        self._mocap_spin_thread = None
        self._mocap_spin_stop = False

        # 在 BasePolicy 初始化前就把本文件创建的 ROS2 subscriber node 加入私有 executor。
        # 后续任何地方都不再直接调用 rclpy.spin_once(node)，而是统一调用 _spin_ros_subscribers_once()。
        self._setup_private_ros_executor()

        super().__init__(config)
        self._ensure_mocap_spin_thread()

        # Load stiff startup parameters from robot config
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            # Fallback to default_dof_angles if not specified
            self._stiff_hold_q = np.array(config.robot.default_dof_angles, dtype=np.float32).reshape(1, -1)

        if config.robot.stiff_startup_kp is not None:
            self._stiff_hold_kp = np.array(config.robot.stiff_startup_kp, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kp for WBT policy")

        if config.robot.stiff_startup_kd is not None:
            self._stiff_hold_kd = np.array(config.robot.stiff_startup_kd, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kd for WBT policy")

        if self._stiff_hold_q.shape[1] != self.num_dofs:
            raise ValueError("Stiff startup pose dimension mismatch with robot DOFs")

        # Prompt user before entering stiff mode (only if stdin is available)
        def _show_warning():
            logger.warning(
                colored(
                    "⚠️  Non-interactive mode detected - cannot prompt for stiff mode confirmation!",
                    "red",
                    attrs=["bold"],
                )
            )

        if sys.stdin.isatty():
            logger.info(colored("\n⚠️  Ready to enter stiff hold mode", "yellow", attrs=["bold"]))
            logger.info(colored("Press Enter to continue...", "yellow"))
            try:
                input()
                logger.info(colored("✓ Entering stiff hold mode", "green"))
            except EOFError:
                # [drockyd] seems like in some cases, input() will raise EOFError even in interactive mode.
                _show_warning()
        else:
            _show_warning()

    @staticmethod
    def _quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
        """与 wbt_sim.get_current_obs_buffer_dict() 内部实现保持一致。"""
        q = q.copy()
        q[:, 0:3] *= -1.0
        return q

    @staticmethod
    def _quat_apply_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """与 wbt_sim 中对齐训练 rotations.py 的 quat_apply(..., w_last=True) 保持一致。

        Parameters
        ----------
        q : np.ndarray
            shape=(N, 4)，xyzw 顺序。
        v : np.ndarray
            shape=(N, 3)。
        """
        q_vec = q[:, 0:3]
        q_w = q[:, 3:4]
        t = 2.0 * np.cross(q_vec, v)
        return v + q_w * t + np.cross(q_vec, t)

    @staticmethod
    def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiplication for xyzw quaternions.

        当前版本主要用于后续如需添加“动捕刚体坐标系 -> 机器人 link 坐标系”的标定修正。
        """
        x1, y1, z1, w1 = q1[:, 0:1], q1[:, 1:2], q1[:, 2:3], q1[:, 3:4]
        x2, y2, z2, w2 = q2[:, 0:1], q2[:, 1:2], q2[:, 2:3], q2[:, 3:4]
        return np.concatenate(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            axis=1,
        ).astype(np.float32)

    def _setup_private_ros_executor(self) -> None:
        """把本策略内部创建的 ROS2 subscriber node 统一加入一个私有 executor。

        中文说明：
        - 不再让不同线程分别调用 rclpy.spin_once(node)；
        - 所有 ROS2 回调都从同一个 SingleThreadedExecutor 里取；
        - 用 self._mocap_spin_lock 保护 executor.spin_once，确保任何时刻只有一个线程在 spin；
        - 这样可以保留 ROS2 接收，同时消除偶发的 "generator already executing"。
        """
        if getattr(self, "_ros_executor_nodes_added", False):
            return

        executor = getattr(self, "_ros_executor", None)
        if executor is None:
            return

        for node in (
            getattr(self, "object_state_sub", None),
            getattr(self, "torso_link_pos_sub", None),
            getattr(self, "basketball_pose_sub", None),
            getattr(self, "waist_pose_sub", None),
        ):
            if node is None:
                continue
            try:
                executor.add_node(node)
            except Exception as exc:
                logger.warning(f"[ROS EXECUTOR] add_node warning for {node}: {exc}")

        self._ros_executor_nodes_added = True

    def _spin_ros_subscribers_once(self, timeout_sec: float = 0.0) -> None:
        """统一处理本策略创建的所有 ROS2 subscriber 回调。

        注意：部署版需要保留 ROS2 接收，但不能在多个线程里同时 spin。
        因此本函数是全文件唯一允许执行 executor.spin_once 的入口。
        """
        executor = getattr(self, "_ros_executor", None)
        lock = getattr(self, "_mocap_spin_lock", None)

        if executor is not None:
            if lock is None:
                executor.spin_once(timeout_sec=float(timeout_sec))
            else:
                with lock:
                    executor.spin_once(timeout_sec=float(timeout_sec))
            return

        # 极端兜底：如果私有 executor 没初始化成功，则仍用同一把锁保护所有 rclpy.spin_once 调用。
        nodes = (
            getattr(self, "object_state_sub", None),
            getattr(self, "torso_link_pos_sub", None),
            getattr(self, "basketball_pose_sub", None),
            getattr(self, "waist_pose_sub", None),
        )
        if lock is None:
            for node in nodes:
                if node is not None:
                    rclpy.spin_once(node, timeout_sec=float(timeout_sec))
        else:
            with lock:
                for node in nodes:
                    if node is not None:
                        rclpy.spin_once(node, timeout_sec=float(timeout_sec))

    def _spin_mocap_subscribers(self) -> None:
        """非阻塞读取一次 ROS2 subscriber 数据。

        历史上这个函数只 spin /basketball/pose 和 /waist/pose。
        现在为了避免 executor 并发问题，它会通过私有 executor 统一处理所有本策略 subscriber，
        包括 /object_state 和 /torso_link_pos。函数名保留是为了兼容原有调用点。
        """
        self._spin_ros_subscribers_once(timeout_sec=0.0)

    def _ensure_mocap_spin_thread(self) -> None:
        """启动常驻动捕接收线程。

        中文说明：
        - 这个线程只负责高频 spin ROS2 mocap subscribers；
        - 不修改 policy/action，不控制机器人；
        - 没有它时，如果只靠每秒一次的 monitor 打印来 spin，会出现 hz≈1、drops/s≈119 的假象。
        """
        thread = getattr(self, "_mocap_spin_thread", None)
        if thread is not None and thread.is_alive():
            return
        self._mocap_spin_stop = False
        self._mocap_spin_thread = threading.Thread(
            target=self._mocap_spin_loop,
            name="wbt_mocap_spin",
            daemon=True,
        )
        self._mocap_spin_thread.start()

    def _mocap_spin_loop(self) -> None:
        """高频处理 /basketball/pose 和 /waist/pose 回调。"""
        last_warn_time = 0.0
        while not getattr(self, "_mocap_spin_stop", False):
            try:
                # 使用私有 executor 统一接收所有 ROS2 subscriber 数据。
                # timeout 用 0，外层 sleep 控制轮询频率，避免控制线程被长时间抢锁。
                self._spin_ros_subscribers_once(timeout_sec=0.0)
            except Exception as exc:
                now = time.monotonic()
                if now - last_warn_time > 1.0:
                    logger.warning(f"[ROS SPIN] subscriber spin warning: {exc}")
                    last_warn_time = now
            time.sleep(float(getattr(self, "mocap_spin_period_sec", 0.001)))

    def _set_real_only_mode(self, enabled: bool, reason: str = "") -> None:
        """切换 real-only / sim-debug 模式。"""
        self.real_only_mode = bool(enabled)
        self.allow_sim_fallback = not self.real_only_mode
        if self.real_only_mode:
            self.logger.warning(
                colored(
                    f"[REAL-ONLY MODE] ENABLED. MuJoCo/sim fallback is FORBIDDEN in all cases. {reason}",
                    "red",
                    attrs=["bold"],
                )
            )
        else:
            self.logger.warning(
                colored(
                    f"[SIM DEBUG MODE] ENABLED. MuJoCo/sim fallback is allowed ONLY for debugging. {reason}",
                    "yellow",
                    attrs=["bold"],
                )
            )

    def _mocap_fresh(self, sub: MocapPoseSub) -> bool:
        return bool(getattr(sub, "received", False)) and sub.age_sec() <= self.mocap_stale_timeout_sec

    def _mocap_ready_now(self) -> bool:
        """当前 basketball 和 waist 动捕是否都已收到且足够新鲜。"""
        self._spin_mocap_subscribers()
        return self._mocap_fresh(self.basketball_pose_sub) and self._mocap_fresh(self.waist_pose_sub)

    def _mocap_new_drop_since_last_check(self) -> bool:
        """检查上次安全门检查之后是否出现新的疑似掉帧。"""
        cur_b = int(getattr(self.basketball_pose_sub, "drop_count", 0))
        cur_w = int(getattr(self.waist_pose_sub, "drop_count", 0))
        prev_b = int(self._mocap_drop_snapshot.get("basketball", 0))
        prev_w = int(self._mocap_drop_snapshot.get("waist", 0))
        new_drop = cur_b > prev_b or cur_w > prev_w
        self._mocap_drop_snapshot = {"basketball": cur_b, "waist": cur_w}
        return new_drop

    def _mocap_health_ok_for_policy(self) -> bool:
        """policy 推理前的真机安全门。"""
        ready = self._mocap_ready_now()
        new_drop = self._mocap_new_drop_since_last_check()
        return ready and not new_drop

    def _source_text_for_mocap_or_sim(self, mocap_sub: MocapPoseSub, sim_received: bool) -> str:
        if self._mocap_fresh(mocap_sub):
            return "mocap"
        if getattr(mocap_sub, "received", False):
            return "mocap_stale"
        if (not self.real_only_mode) and sim_received:
            return "sim_available"
        return "none_no_sim_in_real_only" if self.real_only_mode else "none"

    def _format_mocap_snapshot(self, name: str, sub: MocapPoseSub) -> str:
        snap = sub.snapshot()
        recv_dt = snap.get("last_receive_dt_sec", snap.get("last_dt_sec", float("nan")))
        recv_dt_text = "nan" if not np.isfinite(recv_dt) else f"{recv_dt:.4f}s"
        stamp_dt = snap.get("last_stamp_dt_sec", float("nan"))
        stamp_dt_text = "nan" if not np.isfinite(stamp_dt) else f"{stamp_dt:.4f}s"
        age = snap["age_sec"]
        age_text = "inf" if not np.isfinite(age) else f"{age:.4f}s"

        if getattr(sub, "received", False):
            pos_text = np.round(sub.latest_pos_w.reshape(-1), 4).tolist()
            quat_text = np.round(sub.latest_quat_xyzw.reshape(-1), 4).tolist()
        else:
            pos_text = "NA"
            quat_text = "NA"

        return (
            f"{name}: topic={snap['topic']} | received={snap['received']} | fresh={self._mocap_fresh(sub)} | "
            f"age={age_text} | msg_count={snap['message_count']} | recv_drop_count={snap['drop_count']} | "
            f"recv_drop_events={snap['drop_event_count']} | invalid_count={snap['invalid_count']} | "
            f"recv_dt={recv_dt_text} | max_recv_dt={snap.get('max_receive_dt_sec', 0.0):.4f}s | "
            f"stamp_dt={stamp_dt_text} | max_stamp_dt={snap.get('max_stamp_dt_sec', 0.0):.4f}s | "
            f"stamp_gap_count={snap.get('stamp_gap_count', 0)} | stamp_jump_back_count={snap.get('stamp_jump_back_count', 0)} | "
            f"last_drop_reason={snap['last_drop_reason']} | last_stamp_warning={snap.get('last_stamp_warning', 'none')} | "
            f"pos_w={pos_text} | quat_xyzw={quat_text}"
        )

    def _print_mocap_status(self, prefix: str = "[MOCAP STATUS]") -> None:
        """打印一次状态，不改变机器人控制状态。"""
        self._spin_mocap_subscribers()
        basketball_source = self._source_text_for_mocap_or_sim(
            self.basketball_pose_sub,
            getattr(self.object_state_sub, "received", False),
        )
        waist_source = self._source_text_for_mocap_or_sim(
            self.waist_pose_sub,
            getattr(self.torso_link_pos_sub, "received", False),
        )
        self.logger.info("\n" + "=" * 110)
        self.logger.info(
            f"{prefix} real_only_mode={self.real_only_mode} | allow_sim_fallback={self.allow_sim_fallback} | "
            f"basketball_source={basketball_source} | waist_source={waist_source} | "
            f"expected_hz={self.mocap_expected_hz:.1f} | expected_dt={self.mocap_expected_dt_sec:.4f}s | "
            f"drop_gap={self.mocap_drop_gap_sec:.4f}s | stale_timeout={self.mocap_stale_timeout_sec:.4f}s | "
            f"spin_thread_alive={getattr(getattr(self, '_mocap_spin_thread', None), 'is_alive', lambda: False)()} | "
            f"spin_period={float(getattr(self, 'mocap_spin_period_sec', 0.0)):.4f}s"
        )
        self.logger.info(self._format_mocap_snapshot("basketball", self.basketball_pose_sub))
        self.logger.info(self._format_mocap_snapshot("waist", self.waist_pose_sub))
        self.logger.info(
            f"sim topics visible but not used in real-only: /object_state_received={getattr(self.object_state_sub, 'received', False)} | "
            f"/torso_link_pos_received={getattr(self.torso_link_pos_sub, 'received', False)}"
        )
        self.logger.info("=" * 110 + "\n")

    def _toggle_mocap_status_monitor(self) -> None:
        """按 i 开关连续状态打印模式。"""
        self.mocap_status_monitor_enabled = not self.mocap_status_monitor_enabled
        self._mocap_status_last_print_time = 0.0
        self._mocap_status_prev_counts = {
            "basketball_msg": int(getattr(self.basketball_pose_sub, "message_count", 0)),
            "waist_msg": int(getattr(self.waist_pose_sub, "message_count", 0)),
            "basketball_drop": int(getattr(self.basketball_pose_sub, "drop_count", 0)),
            "waist_drop": int(getattr(self.waist_pose_sub, "drop_count", 0)),
        }
        state = "ON" if self.mocap_status_monitor_enabled else "OFF"
        self.logger.warning(colored(f"[MOCAP MONITOR] {state}. Press 'i' again to toggle.", "cyan"))
        if self.mocap_status_monitor_enabled:
            self._ensure_mocap_monitor_thread()
        self._print_mocap_status(prefix="[MOCAP MONITOR STATUS]")

    def _ensure_mocap_monitor_thread(self) -> None:
        """确保 i 键监测线程已启动。

        中文说明：
        这个线程只负责每秒 spin 一次动捕 subscriber 并打印状态，
        不会修改 policy、action 或机器人控制状态。
        """
        thread = getattr(self, "_mocap_monitor_thread", None)
        if thread is not None and thread.is_alive():
            return
        self._mocap_monitor_stop = False
        self._mocap_monitor_thread = threading.Thread(
            target=self._mocap_monitor_loop,
            name="wbt_mocap_monitor",
            daemon=True,
        )
        self._mocap_monitor_thread.start()

    def _mocap_monitor_loop(self) -> None:
        """每秒统计/打印动捕状态。

        注意：真正的 mocap 接收由 _mocap_spin_loop 常驻线程完成；
        这里不再承担高频接收职责，只负责低频打印。
        """
        while not getattr(self, "_mocap_monitor_stop", False):
            if getattr(self, "mocap_status_monitor_enabled", False):
                try:
                    self._maybe_print_mocap_monitor()
                except Exception as exc:
                    self.logger.warning(f"[MOCAP MONITOR] monitor thread warning: {exc}")
            time.sleep(0.05)

    def _maybe_print_mocap_monitor(self) -> None:
        """i 模式下，每秒打印消息频率和掉帧次数。"""
        if not getattr(self, "mocap_status_monitor_enabled", False):
            return
        now = time.monotonic()
        last = float(getattr(self, "_mocap_status_last_print_time", 0.0))
        if last > 0.0 and now - last < self.mocap_status_print_period_sec:
            return

        self._spin_mocap_subscribers()
        elapsed = self.mocap_status_print_period_sec if last <= 0.0 else max(now - last, 1e-6)
        prev = self._mocap_status_prev_counts
        b_msg = int(getattr(self.basketball_pose_sub, "message_count", 0))
        w_msg = int(getattr(self.waist_pose_sub, "message_count", 0))
        b_drop = int(getattr(self.basketball_pose_sub, "drop_count", 0))
        w_drop = int(getattr(self.waist_pose_sub, "drop_count", 0))

        b_msg_rate = (b_msg - int(prev.get("basketball_msg", b_msg))) / elapsed
        w_msg_rate = (w_msg - int(prev.get("waist_msg", w_msg))) / elapsed
        b_drop_rate = (b_drop - int(prev.get("basketball_drop", b_drop))) / elapsed
        w_drop_rate = (w_drop - int(prev.get("waist_drop", w_drop))) / elapsed

        self._mocap_status_prev_counts = {
            "basketball_msg": b_msg,
            "waist_msg": w_msg,
            "basketball_drop": b_drop,
            "waist_drop": w_drop,
        }
        self._mocap_status_last_print_time = now

        level = self.logger.warning if (b_drop_rate > 0.0 or w_drop_rate > 0.0) else self.logger.info
        level(
            "[MOCAP MONITOR 1s] "
            f"basketball: source={self._source_text_for_mocap_or_sim(self.basketball_pose_sub, getattr(self.object_state_sub, 'received', False))}, "
            f"fresh={self._mocap_fresh(self.basketball_pose_sub)}, hz≈{b_msg_rate:.1f}, recv_drops/s={b_drop_rate:.1f}, "
            f"age={self.basketball_pose_sub.age_sec():.4f}s, recv_dt={getattr(self.basketball_pose_sub, 'last_receive_dt_sec', None)}, "
            f"stamp_dt={getattr(self.basketball_pose_sub, 'last_stamp_dt_sec', None)}, stamp_gap_count={getattr(self.basketball_pose_sub, 'stamp_gap_count', 0)}; "
            f"waist: source={self._source_text_for_mocap_or_sim(self.waist_pose_sub, getattr(self.torso_link_pos_sub, 'received', False))}, "
            f"fresh={self._mocap_fresh(self.waist_pose_sub)}, hz≈{w_msg_rate:.1f}, recv_drops/s={w_drop_rate:.1f}, "
            f"age={self.waist_pose_sub.age_sec():.4f}s, recv_dt={getattr(self.waist_pose_sub, 'last_receive_dt_sec', None)}, "
            f"stamp_dt={getattr(self.waist_pose_sub, 'last_stamp_dt_sec', None)}, stamp_gap_count={getattr(self.waist_pose_sub, 'stamp_gap_count', 0)}"
        )

    def _preflight_mocap_check(self, duration_sec: float | None = None) -> bool:
        """执行 policy 前的 5 秒动捕安全检查。"""
        duration = self.mocap_preflight_duration_sec if duration_sec is None else float(duration_sec)
        self.logger.warning(
            colored(
                f"[MOCAP PREFLIGHT] Checking /basketball/pose and /waist/pose for {duration:.1f}s...",
                "yellow",
                attrs=["bold"],
            )
        )

        self._spin_mocap_subscribers()
        start_time = time.monotonic()
        start_b_msg = int(getattr(self.basketball_pose_sub, "message_count", 0))
        start_w_msg = int(getattr(self.waist_pose_sub, "message_count", 0))
        start_b_drop = int(getattr(self.basketball_pose_sub, "drop_count", 0))
        start_w_drop = int(getattr(self.waist_pose_sub, "drop_count", 0))
        start_b_invalid = int(getattr(self.basketball_pose_sub, "invalid_count", 0))
        start_w_invalid = int(getattr(self.waist_pose_sub, "invalid_count", 0))

        fail_reasons: list[str] = []
        last_print_second = -1
        while time.monotonic() - start_time < duration:
            self._spin_mocap_subscribers()
            elapsed = time.monotonic() - start_time
            cur_second = int(elapsed)
            if cur_second != last_print_second:
                last_print_second = cur_second
                self.logger.info(
                    f"[MOCAP PREFLIGHT] t={elapsed:.1f}/{duration:.1f}s | "
                    f"basketball_fresh={self._mocap_fresh(self.basketball_pose_sub)} age={self.basketball_pose_sub.age_sec():.4f}s | "
                    f"waist_fresh={self._mocap_fresh(self.waist_pose_sub)} age={self.waist_pose_sub.age_sec():.4f}s"
                )

            if not self._mocap_fresh(self.basketball_pose_sub):
                fail_reasons.append(f"basketball_not_fresh_age={self.basketball_pose_sub.age_sec():.4f}s")
                break
            if not self._mocap_fresh(self.waist_pose_sub):
                fail_reasons.append(f"waist_not_fresh_age={self.waist_pose_sub.age_sec():.4f}s")
                break
            if int(getattr(self.basketball_pose_sub, "drop_count", 0)) > start_b_drop:
                fail_reasons.append(f"basketball_receive_drop_detected: {self.basketball_pose_sub.last_drop_reason}")
                break
            if int(getattr(self.waist_pose_sub, "drop_count", 0)) > start_w_drop:
                fail_reasons.append(f"waist_receive_drop_detected: {self.waist_pose_sub.last_drop_reason}")
                break
            if int(getattr(self.basketball_pose_sub, "invalid_count", 0)) > start_b_invalid:
                fail_reasons.append("basketball_invalid_quaternion_detected")
                break
            if int(getattr(self.waist_pose_sub, "invalid_count", 0)) > start_w_invalid:
                fail_reasons.append("waist_invalid_quaternion_detected")
                break
            time.sleep(self.mocap_preflight_poll_sec)

        end_b_msg = int(getattr(self.basketball_pose_sub, "message_count", 0))
        end_w_msg = int(getattr(self.waist_pose_sub, "message_count", 0))
        if end_b_msg <= start_b_msg:
            fail_reasons.append("basketball_no_new_message_during_preflight")
        if end_w_msg <= start_w_msg:
            fail_reasons.append("waist_no_new_message_during_preflight")

        self._mocap_drop_snapshot = {
            "basketball": int(getattr(self.basketball_pose_sub, "drop_count", 0)),
            "waist": int(getattr(self.waist_pose_sub, "drop_count", 0)),
        }

        if fail_reasons:
            self.logger.error(colored(f"[MOCAP PREFLIGHT FAILED] {fail_reasons}", "red", attrs=["bold"]))
            self._print_mocap_status(prefix="[MOCAP PREFLIGHT FAILED STATUS]")
            return False

        self.logger.info(colored("[MOCAP PREFLIGHT PASSED] No drop/stale/invalid data detected.", "green"))
        self._print_mocap_status(prefix="[MOCAP PREFLIGHT PASSED STATUS]")
        return True

    def _hold_current_action_due_to_mocap_loss(self) -> np.ndarray:
        """动捕异常时保持上一帧安全动作，绝不切换 sim 数据。"""
        if self._last_safe_scaled_policy_action is not None:
            hold_action = self._last_safe_scaled_policy_action.copy()
        elif hasattr(self, "scaled_policy_action") and self.scaled_policy_action is not None:
            hold_action = np.asarray(self.scaled_policy_action, dtype=np.float32).copy()
        else:
            hold_action = np.zeros((1, self.num_dofs), dtype=np.float32)

        if not self._holding_due_to_mocap_loss:
            self.logger.error(
                colored(
                    "[MOCAP LOST] Holding last action. Policy inference is paused. Sim/MuJoCo fallback is forbidden.",
                    "red",
                    attrs=["bold"],
                )
            )
            self._print_mocap_status(prefix="[MOCAP LOST STATUS]")
        self._holding_due_to_mocap_loss = True
        return hold_action


    def _get_torso_pose_from_waist_mocap(
        self,
        spin_once: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """由 /waist/pose 的动捕 6D 位姿估计 torso_link 的世界系位姿。

        返回值与 _get_ref_body_pose_in_world() 保持一致：
        - torso_pos_w: shape=(1, 3)，单位 m；
        - torso_quat_wxyz: shape=(1, 4)，wxyz 顺序。

        中文说明：
        这里严格使用和 wbt_sim 里 obj_pos_b 相同的 xyzw 四元数旋转函数。
        PoseStamped.orientation 是 ROS 标准 xyzw，所以先在 xyzw 下做 offset 旋转，
        最后仅在返回给原 WBT 逻辑时转换成 wxyz。
        """
        if spin_once:
            self._spin_mocap_subscribers()

        if not getattr(self.waist_pose_sub, "received", False):
            return None

        waist_pos_w = self.waist_pose_sub.latest_pos_w.astype(np.float32).reshape(1, 3)
        waist_quat_xyzw = self.waist_pose_sub.latest_quat_xyzw.astype(np.float32).reshape(1, 4)

        # p_torso_w = p_waist_w + R_waist_w * p_torso_in_waist
        # 其中 p_torso_in_waist = [0, 0, 0.034] m。
        torso_offset_w = self._quat_apply_xyzw(
            waist_quat_xyzw,
            self.torso_offset_in_waist.astype(np.float32).reshape(1, 3),
        )
        torso_pos_w = (waist_pos_w + torso_offset_w).astype(np.float32).reshape(1, 3)

        # 你给的图里 torso_link 与 waist_yaw_link 坐标轴近似一致，
        # 因此 torso_link 姿态直接采用 waist 动捕刚体姿态。
        torso_quat_wxyz = xyzw_to_wxyz(waist_quat_xyzw).astype(np.float32).reshape(1, 4)
        return torso_pos_w, torso_quat_wxyz


    def _get_ref_body_pose_in_world(self, robot_state_data) -> tuple[np.ndarray, np.ndarray]:
        """基于当前机器人状态，求参考刚体在世界系下的位置和姿态。"""
        # Create configuration for pinocchio robot
        # Note:
        # 1. pinocchio quaternion is in xyzw format, robot_state_data is in wxyz format
        # 2. joint sequences in pinocchio robot and real robot are different

        # free base pos / ori:
        # 实机动捕模式：优先用 /waist/pose 估计 torso_link 的完整 6D 世界位姿。
        # 这样 get_current_obs_buffer_dict() 里后续的 obj_pos_b 计算，
        # 仍然可以完全沿用 wbt_sim 的四元数转换逻辑。
        mocap_torso_pose = self._get_torso_pose_from_waist_mocap(spin_once=True)
        if mocap_torso_pose is not None and self._mocap_fresh(self.waist_pose_sub):
            return mocap_torso_pose

        if getattr(self, "real_only_mode", False):
            raise NoFreshMocapError(
                "REAL-ONLY mode is enabled, but /waist/pose mocap is missing or stale. "
                "Refusing to use MuJoCo/sim torso data."
            )

        # 仿真/兜底模式：保持 wbt_sim 的原始逻辑。
        # 优先使用 MuJoCo 通过 ROS2 发布的 torso_link 世界位置，避免 low_state 中根平移恒为 0。
        self._spin_ros_subscribers_once(timeout_sec=0.0)
        if self.torso_link_pos_sub.received:
            root_pos = self.torso_link_pos_sub.latest_pos_w[0].astype(np.float32)
        else:
            root_pos = robot_state_data[0, :3]
        #print("root_pos",root_pos)

        # free base ori, wxyz -> xyzw
        root_ori_xyzw = wxyz_to_xyzw(robot_state_data[:, 3:7])[0]

        # dof pos in real robot -> pinocchio robot
        num_dofs = self.num_dofs
        dof_pos_in_real = robot_state_data[0, 7 : 7 + num_dofs]
        dof_pos_in_pinocchio = dof_pos_in_real[self.pinocchio_robot.real2pinocchio_index]

        configuration = np.concatenate([root_pos, root_ori_xyzw, dof_pos_in_pinocchio], axis=0)

        ref_pos_w, ref_ori_xyzw = self.pinocchio_robot.fk_and_get_ref_body_pose_in_world(configuration)
        return ref_pos_w, xyzw_to_wxyz(ref_ori_xyzw)

    def _get_ref_body_orientation_in_world(self, robot_state_data):
        _, ref_quat_wxyz = self._get_ref_body_pose_in_world(robot_state_data)
        return ref_quat_wxyz

    def _init_manual_object_reset(self) -> None:
        """初始化球的“按 ] 时复位”逻辑。

        用户给的是球在 motion 第0帧时的世界系位姿；但真正复位时只能用相对位姿，
        否则机器人放下来后的绝对位置一变，球的位置就不对。
        所以这里会：
        1. 读取第0帧参考刚体在世界系下的位姿（来自 npz）
        2. 用用户给的球世界系位姿，换算成“相对参考刚体”的位姿
        3. 按 ] 时再根据当前机器人的参考刚体位姿，把球放回对应世界位置
        """
        self.enable_object_reset_on_start_policy = False
        self.object_init_rel_pos_ref = None
        self.object_init_rel_quat_ref_wxyz = None
        self.object_reset_ref_body_name = None

        motion_cfg = getattr(self.config.robot, "motion", {}) or {}
        self.object_reset_ref_body_name = motion_cfg.get("body_name_ref", [None])[0]
        if self.object_reset_ref_body_name is None:
            self.logger.warning("Object reset disabled: config.robot.motion['body_name_ref'] is missing.")
            return

        # 允许你手动在配置里填世界系初始位姿；如果没填，则回退到 npz 第0帧 object pose。
        manual_object_pos_w = motion_cfg.get("object_init_world_pos", None)
        manual_object_quat_wxyz = (
            motion_cfg.get("object_init_world_quat_wxyz", None)
            or motion_cfg.get("object_init_world_quat", None)
        )

        # 兼容几种可能的 npz 路径字段名
        npz_path = (
            motion_cfg.get("object_pose_ref_npz_path", None)
            or motion_cfg.get("motion_npz_path", None)
            or motion_cfg.get("reference_npz_path", None)
            or motion_cfg.get("npz_path", None)
        )
        if npz_path is None:
            self.logger.warning(
                "Object reset disabled: no npz path found. "
                "Please provide config.robot.motion['object_pose_ref_npz_path']."
            )
            return

        npz_path = str(npz_path)
        if not Path(npz_path).exists():
            self.logger.warning(f"Object reset disabled: npz path does not exist: {npz_path}")
            return

        motion_npz = np.load(npz_path, allow_pickle=True)
        if "body_names" not in motion_npz or "body_pos_w" not in motion_npz:
            self.logger.warning("Object reset disabled: npz is missing body_names/body_pos_w.")
            return

        body_names = [str(name) for name in motion_npz["body_names"].tolist()]
        if self.object_reset_ref_body_name not in body_names:
            self.logger.warning(
                f"Object reset disabled: reference body '{self.object_reset_ref_body_name}' is not in npz body_names."
            )
            return

        ref_body_idx = body_names.index(self.object_reset_ref_body_name)
        ref_pos0_w = np.asarray(motion_npz["body_pos_w"][0, ref_body_idx], dtype=np.float64)

        if "body_quat_w" in motion_npz:
            ref_quat0_wxyz = np.asarray(motion_npz["body_quat_w"][0, ref_body_idx], dtype=np.float64)
        else:
            self.logger.warning("Object reset disabled: npz is missing body_quat_w.")
            return

        if manual_object_pos_w is None:
            if "object_pos_w" in motion_npz:
                manual_object_pos_w = motion_npz["object_pos_w"][0]
                self.logger.info("object_init_world_pos not set; falling back to npz['object_pos_w'][0].")
            else:
                self.logger.warning(
                    "Object reset disabled: neither config.robot.motion['object_init_world_pos'] "
                    "nor npz['object_pos_w'] is available."
                )
                return

        if manual_object_quat_wxyz is None:
            if "object_quat_w" in motion_npz:
                manual_object_quat_wxyz = motion_npz["object_quat_w"][0]
                self.logger.info("object_init_world_quat not set; falling back to npz['object_quat_w'][0].")
            else:
                self.logger.warning(
                    "Object reset disabled: neither config.robot.motion['object_init_world_quat_wxyz'] "
                    "nor npz['object_quat_w'] is available."
                )
                return

        object_pos0_w = np.asarray(manual_object_pos_w, dtype=np.float64).reshape(3)
        object_quat0_wxyz = np.asarray(manual_object_quat_wxyz, dtype=np.float64).reshape(4)

        ref_rot0_w = self._quat_to_rotmat_wxyz(ref_quat0_wxyz)
        self.object_init_rel_pos_ref = ref_rot0_w.T @ (object_pos0_w - ref_pos0_w)
        self.object_init_rel_quat_ref_wxyz = self._quat_mul_wxyz(
            self._quat_conjugate_wxyz(ref_quat0_wxyz),
            object_quat0_wxyz,
        )
        self.enable_object_reset_on_start_policy = True

        self.logger.info(
            colored(
                f"Object reset enabled. Ref body: {self.object_reset_ref_body_name}, "
                f"rel_pos={self.object_init_rel_pos_ref.round(4).tolist()}",
                "green",
            )
        )

    def _compose_object_world_pose_from_current_ref(
        self,
        current_ref_pos_w: np.ndarray,
        current_ref_quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """根据当前参考刚体位姿，重建球的目标世界位姿。"""
        if self.object_init_rel_pos_ref is None or self.object_init_rel_quat_ref_wxyz is None:
            raise RuntimeError("Object relative pose is not initialized yet.")

        ref_rot_w = self._quat_to_rotmat_wxyz(current_ref_quat_wxyz)
        target_pos_w = current_ref_pos_w + ref_rot_w @ self.object_init_rel_pos_ref
        target_quat_wxyz = self._quat_mul_wxyz(current_ref_quat_wxyz, self.object_init_rel_quat_ref_wxyz)
        return target_pos_w, target_quat_wxyz

    def _resolve_simulator_handle(self):
        """尽量从 interface / env 包装层里找到仿真器对象。"""
        candidate_paths = [
            ("interface", "simulator"),
            ("interface", "sim"),
            ("interface", "env", "simulator"),
            ("interface", "env", "sim"),
            ("interface", "env", "unwrapped", "simulator"),
            ("interface", "env", "unwrapped", "sim"),
            ("interface", "_env", "simulator"),
            ("interface", "_env", "sim"),
        ]

        for path in candidate_paths:
            obj = self
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok and obj is not None:
                return obj
        return None

    def _set_object_pose_in_sim(self, pos_w: np.ndarray, quat_wxyz: np.ndarray) -> bool:
        """把球放到仿真中的目标世界位姿。"""
        pos_w = np.asarray(pos_w, dtype=np.float64).reshape(3)
        quat_wxyz = self._quat_normalize_wxyz(quat_wxyz)

        # 方案1：优先调用 MuJoCo 专门暴露出来的公开接口
        if hasattr(self.interface, "set_object_pose_world"):
            self.interface.set_object_pose_world(pos_w, quat_wxyz)
            return True

        sim = self._resolve_simulator_handle()
        if sim is None:
            self.logger.warning("Could not find simulator handle from InterfaceWrapper; object reset skipped.")
            return False

        if hasattr(sim, "set_object_pose_world"):
            sim.set_object_pose_world(pos_w, quat_wxyz, snap_anchor_to_object=True, grab_tether=True)
            return True

        # 方案2：退回到通用 actor 接口
        if hasattr(sim, "get_actor_indices") and hasattr(sim, "set_actor_states_by_index"):
            if torch is None:
                self.logger.warning("torch is unavailable, cannot use generic actor interface to set object pose.")
                return False

            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
            state = np.concatenate(
                [
                    pos_w.astype(np.float32),
                    quat_xyzw,
                    np.zeros(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                ],
                axis=0,
            ).reshape(1, 13)

            device = getattr(sim, "sim_device", "cpu")
            actor_indices = sim.get_actor_indices("object")
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            sim.set_actor_states_by_index(actor_indices, state_tensor, write_updates=True)
            return True

        self.logger.warning("Simulator does not expose a supported object pose setter; object reset skipped.")
        return False

    # def _place_object_to_manual_init_pose(self) -> None:
    #     """按 ] 时，把球放到“当前机器人姿态对应”的初始位置。"""
    #     if not self.enable_object_reset_on_start_policy:
    #         return

    #     robot_state_data = self.interface.get_low_state()
    #     if robot_state_data is None:
    #         self.logger.warning("Object reset skipped: failed to read robot state.")
    #         return

    #     current_ref_pos_w, current_ref_quat_wxyz = self._get_ref_body_pose_in_world(robot_state_data)
    #     target_pos_w, target_quat_wxyz = self._compose_object_world_pose_from_current_ref(
    #         np.asarray(current_ref_pos_w[0], dtype=np.float64),
    #         np.asarray(current_ref_quat_wxyz[0], dtype=np.float64),
    #     )

    #     if self._set_object_pose_in_sim(target_pos_w, target_quat_wxyz):
    #         self.logger.info(
    #             colored(
    #                 f"Object reset on ] succeeded. "
    #                 f"target_pos_w={target_pos_w.round(4).tolist()}",
    #                 "green",
    #             )
    #         )

    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        # Extract KP/KD from ONNX metadata (same as base class)
        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        # Extract URDF text from ONNX metadata
        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            from pathlib import Path

            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        # get initial command and ref quat xyzw
        time_step = np.zeros((1, 1), dtype=np.float32)

        # Use configured observation dimensions (including history) instead of a hard-coded value.
        actor_obs_template = self.obs_buf_dict.get("actor_obs")
        if actor_obs_template is None:
            raise ValueError("Observation group 'actor_obs' must be configured for WBT policy.")
        obs = actor_obs_template.copy()
        input_feed = {"obs": obs, "time_step": time_step}
        outputs = self.onnx_policy_session.run(["joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)

        # motion_command_t/ref_quat_xyzw_t will be used in get_current_obs_buffer_dict
        self.motion_command_t = np.concatenate(outputs[0:2], axis=1)  # (1, 58)
        self.ref_quat_xyzw_t = outputs[2]
        # duplicate, will be used in _get_init_target and _handle_stop_policy
        self.motion_command_0 = self.motion_command_t.copy()
        self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()

        def policy_act(input_feed):
            output = self.onnx_policy_session.run(["actions", "joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)
            action = output[0]
            motion_command = np.concatenate(output[1:3], axis=1)
            ref_quat_xyzw = output[3]
            return action, motion_command, ref_quat_xyzw

        self.policy = policy_act

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "motion_command_0": self.motion_command_0.copy(),
                "ref_quat_xyzw_0": self.ref_quat_xyzw_0.copy(),
            }
        )
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self.motion_command_0 = state["motion_command_0"].copy()
        self.ref_quat_xyzw_0 = state["ref_quat_xyzw_0"].copy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self.robot_yaw_offset = 0.0

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            # Interpolate from current dof_pos to first pose in motion command
            target_dof_pos = self.motion_command_0[:, : self.num_dofs]

            q_target = dof_pos + (target_dof_pos - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    # def get_current_obs_buffer_dict(self, robot_state_data):
    #     current_obs_buffer_dict = {}

    #     # motion_command
    #     current_obs_buffer_dict["motion_command"] = self.motion_command_t

    #     # motion_ref_ori_b
    #     motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)  # wxyz
    #     motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

    #     # robot_ref_ori
    #     robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  #  wxyz
    #     robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)

    #     motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
    #     current_obs_buffer_dict["motion_ref_ori_b"] = motion_ref_ori_b[..., :2].reshape(1, -1)

    #     # base_ang_vel
    #     current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]

    #     # dof_pos
    #     current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles

    #     # dof_vel
    #     current_obs_buffer_dict["dof_vel"] = robot_state_data[
    #         :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
    #     ]

    #     # actions
    #     current_obs_buffer_dict["actions"] = self.last_policy_action

    #     return current_obs_buffer_dict
    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = {}

        # motion_command
        current_obs_buffer_dict["motion_command"] = self.motion_command_t

        # motion_ref_ori_b
        motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)  # wxyz
        motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

        # robot_ref_ori
        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  # wxyz
        robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)
        motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
        current_obs_buffer_dict["motion_ref_ori_b"] = motion_ref_ori_b[..., :2].reshape(1, -1)

        # base_ang_vel
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6
        ]

        # dof_pos
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles

        # dof_vel
        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        # actions
        current_obs_buffer_dict["actions"] = self.last_policy_action

        # 有关物体的：
        # - real-only 模式：必须使用 /basketball/pose + /waist/pose，绝不切换到 MuJoCo/sim；
        # - sim-debug 模式：只有显式按 m 进入后，才允许 fallback 到 wbt_sim 原始逻辑。
        self._spin_mocap_subscribers()
        use_mocap_object = self._mocap_fresh(self.basketball_pose_sub) and self._mocap_fresh(self.waist_pose_sub)

        if use_mocap_object:
            obj_source = "mocap"
            obj_pos_w = self.basketball_pose_sub.latest_pos_w.astype(np.float32).reshape(1, 3)
            torso_pose = self._get_torso_pose_from_waist_mocap(spin_once=False)
            if torso_pose is None:
                raise NoFreshMocapError("/waist/pose mocap disappeared unexpectedly after freshness check.")
            robot_ref_pos_w, robot_ref_quat_wxyz = torso_pose
        else:
            if getattr(self, "real_only_mode", False):
                raise NoFreshMocapError(
                    "REAL-ONLY mode is enabled, but /basketball/pose and /waist/pose are not both fresh. "
                    "Refusing to use MuJoCo/sim object or torso data."
                )

            obj_source = "mujoco"
            if not self._mocap_wait_warned:
                self.logger.warning(
                    "Mocap /basketball/pose and /waist/pose are not both fresh; "
                    "falling back to wbt_sim /object_state + /torso_link_pos logic because SIM DEBUG mode allows it."
                )
                self._mocap_wait_warned = True

            # 仿真/兜底模式：保持你给的 wbt 版本的原始逻辑。注意：real-only 模式下绝不会走到这里。
            self._spin_ros_subscribers_once(timeout_sec=0.0)
            obj_pos_w = self.object_state_sub.latest_pos_w.astype(np.float32).reshape(1, 3)
            robot_ref_pos_w, robot_ref_quat_wxyz = self._get_ref_body_pose_in_world(robot_state_data)

        robot_ref_pos_w = robot_ref_pos_w.astype(np.float32).reshape(1, 3)
        robot_ref_quat_xyzw = wxyz_to_xyzw(robot_ref_quat_wxyz).astype(np.float32).reshape(1, 4)

        # 下面这三行完全照搬你给的 wbt 版本中的 obj_pos_b 相对位姿计算方式：
        # q10 = quat_inverse(q01, w_last=True)
        # t12 = quat_apply(q10, t02 - t01, w_last=True)
        q10_xyzw = self._quat_conjugate_xyzw(robot_ref_quat_xyzw)
        obj_pos_b = self._quat_apply_xyzw(q10_xyzw, obj_pos_w - robot_ref_pos_w).astype(np.float32)

        #current_obs_buffer_dict["obj_pos_b"] = obj_pos_b.reshape(1, 3)
        current_obs_buffer_dict["obj_pos_b"] = np.array([0.43970000743865967,0.008999999612509809,0.37279999256134033])
        if getattr(self, "debug_obj_pos_b_source_enabled", False):
            self._debug_obj_pos_b_counter += 1
            every_n = max(int(getattr(self, "debug_obj_pos_b_every_n", 50)), 1)
            source_changed = obj_source != getattr(self, "_last_obj_pos_b_source", None)
            if source_changed or self._debug_obj_pos_b_counter % every_n == 0:
                self.logger.info(
                    "[OBJ_POS_B DEBUG] "
                    f"source={obj_source} | real_only={self.real_only_mode} | "
                    f"use_mocap_object={use_mocap_object} | "
                    f"basketball_fresh={self._mocap_fresh(self.basketball_pose_sub)} | "
                    f"waist_fresh={self._mocap_fresh(self.waist_pose_sub)} | "
                    f"basketball_age={self.basketball_pose_sub.age_sec():.4f}s | "
                    f"waist_age={self.waist_pose_sub.age_sec():.4f}s | "
                    f"obj_pos_b={np.round(obj_pos_b.reshape(-1), 4).tolist()} | "
                    f"obj_pos_w={np.round(obj_pos_w.reshape(-1), 4).tolist()} | "
                    f"ref_pos_w={np.round(robot_ref_pos_w.reshape(-1), 4).tolist()}"
                )
                self._last_obj_pos_b_source = obj_source

        return current_obs_buffer_dict

    def rl_inference(self, robot_state_data):
        # i 模式下持续打印动捕健康状态，不改变控制命令。
        self._maybe_print_mocap_monitor()

        # prepare obs, run policy inference
        if not self.motion_clip_progressing:
            # Keep motion index pinned at the start while waiting to trigger the clip.
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None

        # real-only 安全门：动捕不新鲜或出现新的疑似掉帧时，绝不 prepare_obs_for_rl，
        # 从源头避免 fallback 到 MuJoCo/sim 数据；直接保持上一帧安全动作。
        if getattr(self, "real_only_mode", False):
            if not self._mocap_health_ok_for_policy():
                return self._hold_current_action_due_to_mocap_loss()
            if self._holding_due_to_mocap_loss:
                self.logger.info(colored("[MOCAP RECOVERED] Fresh mocap received. Policy inference resumes.", "green"))
                self._holding_due_to_mocap_loss = False

        try:
            obs = self.prepare_obs_for_rl(robot_state_data)
        except NoFreshMocapError as exc:
            self.logger.error(colored(f"[REAL-ONLY SAFETY BLOCK] {exc}", "red", attrs=["bold"]))
            return self._hold_current_action_due_to_mocap_loss()

        #print('obs["actor_obs"]',obs["actor_obs"])
        input_feed = {"time_step": np.array([[self.motion_timestep]], dtype=np.float32), "obs": obs["actor_obs"]}
        policy_action, self.motion_command_t, self.ref_quat_xyzw_t = self.policy(input_feed)

        # clip policy action
        policy_action = np.clip(policy_action, -100, 100)
        # store last policy action
        self.last_policy_action = policy_action.copy()
        # scale policy action
        self.scaled_policy_action = policy_action * self.policy_action_scale
        self._last_safe_scaled_policy_action = self.scaled_policy_action.copy()

        # update motion timestep
        if self.motion_clip_progressing:
            if self.use_sim_time:
                self._update_clock()
            else:
                self.motion_timestep += 1
        return self.scaled_policy_action

    def _get_manual_command(self, robot_state_data):
        # TODO: instead of adding kp/kd_override in def _set_motor_command,
        # just use the motor_kp/motor_kd when calling it in _fill_motor_commands
        if not self._stiff_hold_active:
            return None
        return {
            "q": self._stiff_hold_q.copy(),
            "kp": self._stiff_hold_kp,
            "kd": self._stiff_hold_kd,
        }

    def _handle_start_policy(self):
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._capture_robot_yaw_offset()
        self._capture_motion_yaw_offset(self.ref_quat_xyzw_0)

    def _update_clock(self):
        # Use synchronized clock with motion-relative timing
        current_clock = self.clock_sub.get_clock()
        if self.motion_start_timestep is None:
            # Motion just started; anchor to the first received clock tick.
            self.motion_start_timestep = current_clock
        elif self._last_clock_reading is not None and current_clock < self._last_clock_reading:
            # Simulator clock jumped backwards (e.g., reset). Re-anchor start time while preserving progress.
            offset_ms = round(self.motion_timestep * self.timestep_interval_ms)
            self.logger.warning("Clock sync returned earlier timestamp; adjusting motion timing anchor.")
            self.motion_start_timestep = current_clock - offset_ms
        self._last_clock_reading = current_clock
        elapsed_ms = current_clock - self.motion_start_timestep
        if self.motion_timestep == 0 and int(elapsed_ms // self.timestep_interval_ms) > 1:
            self.logger.warning(
                "Still at the beginning but the clock jumped ahead: elapsed_ms={elapsed_ms}, self.timestep_interval_ms="
                "{timestep_interval_ms}, self.motion_timestep={motion_timestep}. "
                "Re-anchoring to the current timestamp so the motion always starts from frame 0.",
                elapsed_ms=elapsed_ms,
                timestep_interval_ms=self.timestep_interval_ms,
                motion_timestep=self.motion_timestep,
            )
            # Still at the beginning but the clock jumped ahead (e.g., due to waiting before start).
            # Re-anchor to the current timestamp so the motion always starts from frame 0.
            self.motion_start_timestep = current_clock
            self._last_clock_reading = current_clock
            self.motion_timestep = 0
            return
        previous_motion_timestep = self.motion_timestep
        self.motion_timestep = int(elapsed_ms // self.timestep_interval_ms)
        if self.motion_timestep != previous_motion_timestep:
            self.logger.info(
                "Motion timestep advanced from {previous_motion_timestep} to {motion_timestep}",
                previous_motion_timestep=previous_motion_timestep,
                motion_timestep=self.motion_timestep,
            )

    def _handle_stop_policy(self):
        """Handle stop policy action."""
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self.logger.info("Actions set to stiff startup command")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None  # Reset motion start time
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_command_t = self.motion_command_0.copy()
        self._last_clock_reading = None
        self.robot_yaw_offset = 0.0

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.clock_sub.reset_origin()
        self.motion_clip_progressing = True
        # Capture motion-specific start timestep for policy-level timing control
        self.motion_start_timestep = None  # will be set in rl_inference
        self.motion_timestep = 0  # Reset to start from beginning of motion
        self._last_clock_reading = None
        self.logger.info(colored("Starting motion clip", "blue"))

    def handle_keyboard_button(self, keycode):
        """Handle keyboard buttons.

        安全相关按键：
        - r：进入 real-only 安全模式，禁止任何 MuJoCo/sim fallback；
        - m：进入 sim-debug 模式，允许 fallback，仅建议仿真调试使用；
        - i：开关 basketball / waist 动捕状态连续打印模式，不做任何机器人控制；
        - ]：执行 policy 前先进行 5 秒动捕 preflight，失败则不把事件交给 BasePolicy。
        """
        key = str(keycode).lower()
        if key == self.real_mode_key:
            self._set_real_only_mode(True, reason="Keyboard 'r' pressed.")
            self._print_mocap_status(prefix="[REAL MODE STATUS]")
            return
        if key == self.sim_mode_key:
            self._set_real_only_mode(False, reason="Keyboard 'm' pressed.")
            self._print_mocap_status(prefix="[SIM DEBUG STATUS]")
            return
        if key == self.mocap_status_key:
            self._toggle_mocap_status_monitor()
            return

        if keycode == "]":
            if getattr(self, "real_only_mode", False):
                if not self._preflight_mocap_check(duration_sec=self.mocap_preflight_duration_sec):
                    self.logger.error("Policy start by keyboard ']' was blocked by mocap preflight failure.")
                    return
            else:
                self.logger.warning(
                    colored(
                        "[SIM DEBUG MODE] Keyboard ']' starts policy without mocap preflight. "
                        "MuJoCo/sim fallback is allowed because keyboard 'm' was pressed.",
                        "yellow",
                    )
                )
            super().handle_keyboard_button(keycode)
            return

        if keycode == "s":
            self.clock_sub.reset_origin()
            self._handle_start_motion_clip()
        else:
            super().handle_keyboard_button(keycode)

    def handle_joystick_button(self, cur_key):
        """Handle joystick / Unitree remote button presses for WBT-specific controls.

        宇树遥控器安全相关组合键：
        - select+X：等价于键盘 i，开关 basketball / waist 动捕状态连续打印，不改变机器人控制状态；
        - select+Y：等价于键盘 r，进入 real-only 真机安全模式，禁止任何 MuJoCo/sim fallback；
        - select+B：等价于键盘 m，进入 sim-debug 模式，允许 MuJoCo/sim fallback，仅用于仿真调试；
        - A：保留原来的 start policy 语义，但 real-only 下会先执行 5 秒 mocap preflight。

        其他原始遥控器逻辑交给 BasePolicy：
        - B：stop policy；
        - Y：init state；
        - select：切换 policy；
        - L1+R1：退出程序。
        """
        key_raw = str(cur_key)
        key = key_raw.lower()

        if getattr(self, "debug_joystick_key_enabled", False):
            self.logger.warning(f"[JOYSTICK DEBUG] cur_key={key_raw!r} | normalized={key!r}")

        if key in getattr(self, "joystick_mocap_status_keys", set()):
            self.logger.warning(
                colored(
                    f"[JOYSTICK] {cur_key} pressed -> toggle mocap monitor, same as keyboard 'i'.",
                    "cyan",
                    attrs=["bold"],
                )
            )
            self._toggle_mocap_status_monitor()
            super()._print_control_status()
            return

        if key in getattr(self, "joystick_real_mode_keys", set()):
            self._set_real_only_mode(True, reason=f"Joystick '{cur_key}' pressed, same as keyboard 'r'.")
            self._print_mocap_status(prefix="[JOYSTICK REAL MODE STATUS]")
            super()._print_control_status()
            return

        if key in getattr(self, "joystick_sim_mode_keys", set()):
            self._set_real_only_mode(False, reason=f"Joystick '{cur_key}' pressed, same as keyboard 'm'.")
            self._print_mocap_status(prefix="[JOYSTICK SIM DEBUG STATUS]")
            super()._print_control_status()
            return

        if key == "a":
            if getattr(self, "real_only_mode", False):
                if not self._preflight_mocap_check(duration_sec=self.mocap_preflight_duration_sec):
                    self.logger.error("Policy start by joystick A was blocked by mocap preflight failure.")
                    super()._print_control_status()
                    return
            else:
                self.logger.warning(
                    colored(
                        "[SIM DEBUG MODE] Joystick A starts policy without mocap preflight. "
                        "MuJoCo/sim fallback is allowed because sim-debug mode is enabled.",
                        "yellow",
                    )
                )
            # 检查通过后，再交给 BasePolicy 的原始 A 逻辑：_handle_start_policy()。
            super().handle_joystick_button(cur_key)
            super()._print_control_status()
            return

        if key == "start":
            # Start playing motion clip
            self._handle_start_motion_clip()
        else:
            # Delegate all other buttons to BasePolicy to preserve original Unitree remote behavior.
            super().handle_joystick_button(cur_key)
        super()._print_control_status()

    def _capture_robot_yaw_offset(self):
        """Capture robot yaw when policy starts to use as reference offset."""
        robot_state_data = self.interface.get_low_state()
        print("robot_state_data",robot_state_data)
        if robot_state_data is None:
            self.robot_yaw_offset = 0.0
            self.logger.warning("Unable to capture robot yaw offset - missing robot state.")
            return

        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  # wxyz
        yaw = self._quat_yaw(robot_ref_ori)
        self.robot_yaw_offset = yaw
        self.logger.info(colored(f"Robot yaw offset captured at {np.degrees(yaw):.1f} deg", "blue"))

    def _capture_motion_yaw_offset(self, ref_quat_xyzw_0: np.ndarray) -> float:
        """Capture motion yaw when policy starts to use as reference offset."""
        self.motion_yaw_offset = self._quat_yaw(xyzw_to_wxyz(ref_quat_xyzw_0))
        self.logger.info(colored(f"Motion yaw offset captured at {np.degrees(self.motion_yaw_offset):.1f} deg", "blue"))

    def _remove_yaw_offset(self, quat_wxyz: np.ndarray, yaw_offset: float) -> np.ndarray:
        """Remove stored yaw offset from robot orientation quaternion."""
        if abs(yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw_offset)).reshape(1, 4)
        yaw_quat = np.broadcast_to(yaw_quat, quat_wxyz.shape)
        return quat_mul(yaw_quat, quat_wxyz)


    def __del__(self):
        try:
            self._mocap_spin_stop = True
            self._mocap_monitor_stop = True
            self.mocap_status_monitor_enabled = False
        except Exception:
            pass
        try:
            spin_thread = getattr(self, "_mocap_spin_thread", None)
            if spin_thread is not None and spin_thread.is_alive():
                spin_thread.join(timeout=0.2)
        except Exception:
            pass
        try:
            monitor_thread = getattr(self, "_mocap_monitor_thread", None)
            if monitor_thread is not None and monitor_thread.is_alive():
                monitor_thread.join(timeout=0.2)
        except Exception:
            pass
        try:
            executor = getattr(self, "_ros_executor", None)
            if executor is not None:
                for node in (
                    getattr(self, "object_state_sub", None),
                    getattr(self, "torso_link_pos_sub", None),
                    getattr(self, "basketball_pose_sub", None),
                    getattr(self, "waist_pose_sub", None),
                ):
                    if node is not None:
                        try:
                            executor.remove_node(node)
                        except Exception:
                            pass
                try:
                    executor.shutdown()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if hasattr(self, "object_state_sub") and self.object_state_sub is not None:
                self.object_state_sub.destroy_node()
        except Exception:
            pass
        try:
            if hasattr(self, "torso_link_pos_sub") and self.torso_link_pos_sub is not None:
                self.torso_link_pos_sub.destroy_node()
        except Exception:
            pass
        try:
            if hasattr(self, "basketball_pose_sub") and self.basketball_pose_sub is not None:
                self.basketball_pose_sub.destroy_node()
        except Exception:
            pass
        try:
            if hasattr(self, "waist_pose_sub") and self.waist_pose_sub is not None:
                self.waist_pose_sub.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    @staticmethod
    def _quat_yaw(quat_wxyz: np.ndarray) -> float:
        """Extract yaw angle from quaternion array of shape (1, 4)."""
        quat_flat = quat_wxyz.reshape(-1, 4)[0]
        _, _, yaw = quat_to_rpy(quat_flat)
        return float(yaw)
