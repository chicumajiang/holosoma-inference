import json
import sys
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
    - 你的截图里 position 数值像是 mm，因此默认可以通过 position_scale=0.001 转成 m。
    """

    def __init__(self, topic_name: str, node_name: str, position_scale: float = 1.0):
        super().__init__(node_name)
        self.topic_name = topic_name
        self.position_scale = float(position_scale)
        self.latest_pos_w = np.zeros((1, 3), dtype=np.float32)
        self.latest_quat_xyzw = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.received = False
        self.create_subscription(PoseStamped, topic_name, self._callback, 10)

    def _callback(self, msg: PoseStamped) -> None:
        pos = msg.pose.position
        quat = msg.pose.orientation

        self.latest_pos_w = (
            np.array([[pos.x, pos.y, pos.z]], dtype=np.float32) * self.position_scale
        )

        quat_xyzw = np.array([[quat.x, quat.y, quat.z, quat.w]], dtype=np.float32)
        norm = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
        if not np.all(np.isfinite(quat_xyzw)) or float(norm[0, 0]) < 1e-8:
            self.get_logger().warning(f"{self.topic_name} quaternion is invalid; keeping previous pose.")
            return

        self.latest_quat_xyzw = quat_xyzw / norm
        self.received = True


class WholeBodyTrackingPolicy(BasePolicy):
    def __init__(self, config: InferenceConfig):
        # initialize timestep
        self.motion_timestep = 0
        self.motion_clip_progressing = False
        self.motion_start_timestep = None
        self.motion_command_t = None
        self.ref_quat_xyzw_t = None
        self.ref_pos_xyz_t = None
        self.current_ref_joint_pos = None
        self.current_ref_joint_vel = None
        self.current_ref_pos_xyz = None
        self.current_ref_quat_xyzw = None
        self.motion_command_0 = None
        self.ref_quat_xyzw_0 = None
        self.ref_pos_xyz_0 = None
        self.motion_yaw_offset = 0.0

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
        # 2) 实机动捕模式优先使用 /basketball/pose 和 /waist/pos；
        # 3) 你的动捕 position 截图看起来是 mm，所以默认 scale=0.001 转成 m。
        self.object_state_sub = ObjectStateSub(topic_name="/object_state")
        self.torso_link_pos_sub = TorsoLinkPosSub(topic_name="/torso_link_pos")
        self.mocap_position_scale = 0.001
        self.basketball_pose_sub = MocapPoseSub(
            topic_name="/basketball/pose",
            node_name="wbt_basketball_pose_subscriber",
            position_scale=self.mocap_position_scale,
        )
        self.waist_pose_sub = MocapPoseSub(
            topic_name="/waist/pose",
            node_name="wbt_waist_pose_subscriber",
            position_scale=self.mocap_position_scale,
        )

        # 中文说明：
        # 不要在键盘线程和控制循环里同时调用 rclpy.spin_once() 的全局 executor，
        # 否则会偶发 ValueError: generator already executing。
        # 这里给 WBT 自己的几个订阅节点单独建立 executor，并用锁保证所有 spin_once 串行执行。
        self._rclpy_spin_lock = threading.Lock()
        self._wbt_executor = SingleThreadedExecutor()
        for _node in (
            self.object_state_sub,
            self.torso_link_pos_sub,
            self.basketball_pose_sub,
            self.waist_pose_sub,
        ):
            self._wbt_executor.add_node(_node)

        # torso_link 相对于 waist_yaw_link 的近似平移。
        # 中文说明：根据 URDF / 最新测量结果，保留 x 方向约 -3.9635 mm 偏移，
        # z 方向使用最新测量值 0.034 m。
        # 该 offset 表达在 waist_yaw_link / waist 动捕刚体坐标系下。
        self.torso_offset_in_waist = np.array([[-0.0039635, 0.0, 0.034]], dtype=np.float32)
        self._mocap_wait_warned = False

        # Debug: print which data source is used for obj_pos_b.
        # 中文说明：
        # - source=mocap 表示当前使用 /basketball/pose + /waist/pos；
        # - source=mujoco 表示当前回退使用 /object_state + /torso_link_pos；
        # - 默认每 debug_obj_pos_b_every_n 次 obs 生成打印一次，并且数据源切换时会立即打印。
        self.debug_obj_pos_b_source_enabled = True
        self.debug_obj_pos_b_every_n = 50
        self._debug_obj_pos_b_counter = 0
        self._last_obj_pos_b_source = None

        # 中文说明：按 s/c 切换投篮/恢复 policy 时，打印当前 ONNX 内置 reference。
        # 这样可以确认当前到底在 tracking 哪个 policy 内部的 reference trajectory。
        self.debug_reference_on_stage_start_enabled = True
        self.debug_reference_joint_preview_count = 6

        # 中文说明：
        # 新恢复 policy 训练时 actor_obs 最后三维 obj_pos_b 恒为 0。
        # 因此从按 c 进入 recovery 准备阶段开始，到 x 启动 recovery tracking 期间，
        # 部署端也必须强制把 obj_pos_b 置零，避免真实球位置造成分布偏移。
        self.force_zero_obj_pos_b = False
        self._zero_obj_pos_b_debug_printed = False

        super().__init__(config)

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

        # 中文说明：恢复 policy 第 0 帧 reference joint_pos。
        # 按 c 后不会立刻运行恢复 policy，而是先用 stiff startup 的 kp/kd 固定到这个姿态；
        # 再按 x 才真正启动恢复 policy tracking。
        self._recovery_ref0_hold_q = np.array([
            0.41115668416023254,
            0.06297832727432251,
            0.31799548864364624,
            -0.08726699650287628,
            -0.0576867051422596,
            -0.06092328950762749,
            0.408286452293396,
            -0.15160797536373138,
            -0.3365215063095093,
            -0.08726699650287628,
            0.026435405015945435,
            0.09956204891204834,
            0.03294156864285469,
            -0.029027780517935753,
            0.23954658210277557,
            -2.313858985900879,
            0.5601935982704163,
            -1.0042846202850342,
            1.365410566329956,
            0.2326342612504959,
            0.03872094675898552,
            -0.2490057498216629,
            -2.2163236141204834,
            -0.4963211417198181,
            0.8253850340843201,
            1.2174348831176758,
            -0.7044774889945984,
            0.09724337607622147,
            0.0812341719865799,
        ], dtype=np.float32).reshape(1, -1)
        if self._recovery_ref0_hold_q.shape[1] != self.num_dofs:
            raise ValueError(
                "Recovery ref0 hold pose dimension mismatch with robot DOFs: "
                f"{self._recovery_ref0_hold_q.shape[1]} vs {self.num_dofs}"
            )

        # 中文说明：两个 policy 各自的 action zero-point / default_q。
        # policy 输出的是 q_action，部署侧目标关节位置必须写成：
        #   q_target = policy_default_q + q_action * action_scale
        # 不能所有 policy 都共用 shooting 的 default_q，否则 recovery policy 会整体错位。
        self._shooting_ref0_default_q = np.array([
            -0.17104841768741608,
            0.09444943070411682,
            0.26210758090019226,
            0.1878250539302826,
            -0.10962207615375519,
            0.0001973411563085392,
            -0.16742774844169617,
            -0.18660227954387665,
            -0.26155975461006165,
            0.1700589507818222,
            0.00045667431550100446,
            0.0711684599518776,
            0.03850064426660538,
            -0.009034798480570316,
            0.09702195972204208,
            -1.183659315109253,
            0.20094215869903564,
            -0.4117770791053772,
            1.0605320930480957,
            -0.3890945613384247,
            -0.10555541515350342,
            0.34226590394973755,
            -1.1783448457717896,
            -0.17074644565582275,
            0.23504137992858887,
            1.054811954498291,
            0.12367469817399979,
            -0.09968511760234833,
            -0.5193578600883484,
        ], dtype=np.float32).reshape(1, -1)
        self._recovery_ref0_default_q = self._recovery_ref0_hold_q.copy()
        if self._shooting_ref0_default_q.shape[1] != self.num_dofs:
            raise ValueError(
                "Shooting ref0 default_q dimension mismatch with robot DOFs: "
                f"{self._shooting_ref0_default_q.shape[1]} vs {self.num_dofs}"
            )
        if self._recovery_ref0_default_q.shape[1] != self.num_dofs:
            raise ValueError(
                "Recovery ref0 default_q dimension mismatch with robot DOFs: "
                f"{self._recovery_ref0_default_q.shape[1]} vs {self.num_dofs}"
            )
        self._manual_hold_target = "stiff_startup"

        # 中文说明：
        # 默认启动回到 shooting policy 的 default_q / action offset。
        # 即启动后 active_policy_index=0，obj_pos_b 正常读取真实球位置；
        # 只有按 c/x/v 进入 recovery 流程后，才切到 recovery default_q 并强制 obj_pos_b=0。
        self.force_zero_obj_pos_b = False
        self._zero_obj_pos_b_debug_printed = False
        self._manual_hold_target = "stiff_startup"

        try:
            if getattr(self, "active_policy_index", 0) != 0:
                self._activate_policy(0)

            # 启动时不 tracking，只保持原始 stiff_startup_pos；
            # 按 ] 后启用 shooting policy，按 s 才开始 shooting tracking。
            self.use_policy_action = False
            self.get_ready_state = False
            self._stiff_hold_active = True
            self.motion_clip_progressing = False
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None

            if hasattr(self, "interface") and hasattr(self.interface, "no_action"):
                self.interface.no_action = 0

            self.logger.info(
                colored(
                    "Default startup mode: shooting policy slot [1], "
                    "using shooting ref0 default_q/action offset. "
                    "Press ] to enable shooting policy, s to start shooting tracking.",
                    "green",
                )
            )
        except Exception as exc:
            self.logger.warning(f"Failed to switch to shooting policy at startup: {exc}")


        # 中文说明：自动 skill 切换使用的 shooting 完成姿态。
        # 该 q_ref 来自上传的 npz: joint_pos[214, 7:]，共 29 个 DOF。
        # 运行 shooting tracking 时，每帧比较当前 q 和该 q_ref 的相似程度；
        # 连续满足阈值后，自动切换到 recovery 流程。
        self.auto_q_switch_enabled = True
        #self.auto_q_switch_target_frame = 214
        # self.auto_q_switch_q_ref = np.array([
        #     0.338236540555954,
        #     0.0552116334438324,
        #     0.2781735062599182,
        #     -0.08726699650287628,
        #     -0.024670470505952835,
        #     -0.02885677106678486,
        #     0.3530007302761078,
        #     -0.16448858380317688,
        #     -0.35545799136161804,
        #     -0.08726699650287628,
        #     0.11103832721710205,
        #     0.13113948702812195,
        #     0.01646254025399685,
        #     -0.04095205292105675,
        #     0.21390646696090698,
        #     -1.555882453918457,
        #     0.5346580743789673,
        #     -0.760523796081543,
        #     1.1910011768341064,
        #     -0.00825558789074421,
        #     0.004551858175545931,
        #     0.0947711318731308,
        #     -1.5334417819976807,
        #     -0.4642290472984314,
        #     0.6446005702018738,
        #     1.0156071186065674,
        #     -0.3381797969341278,
        #     -0.054411791265010834,
        #     -0.3028491735458374,
        # ], dtype=np.float32).reshape(1, -1)
        self.auto_q_switch_q_ref = np.array([
    0.34319984912872314,
    0.05570460855960846,
    0.2788521349430084,
    -0.08726699650287628,
    -0.02501346543431282,
    -0.03092554770410061,
    0.3587481677532196,
    -0.16459757089614868,
    -0.3593273162841797,
    -0.08726699650287628,
    0.11093936115503311,
    0.13184921443462372,
    0.018828708678483963,
    -0.04159209132194519,
    0.22041019797325134,
    -1.6007049083709717,
    0.5399448871612549,
    -0.761478841304779,
    1.1955320835113525,
    -0.020575454458594322,
    0.006526161916553974,
    0.08634626120328903,
    -1.5857428312301636,
    -0.47534358501434326,
    0.6491610407829285,
    1.010140299797058,
    -0.34406936168670654,
    -0.04607175663113594,
    -0.29990628361701965,
        ], dtype=np.float32).reshape(1, -1)
        if self.auto_q_switch_q_ref.shape[1] != self.num_dofs:
            raise ValueError(
                "Auto q-switch reference dimension mismatch with robot DOFs: "
                f"{self.auto_q_switch_q_ref.shape[1]} vs {self.num_dofs}"
            )

        # 阈值说明：
        # - mean_abs_err: 29 个关节平均绝对误差，越小越像指定完成帧；
        # - max_abs_err: 单个关节最大误差，避免某个关节偏差特别大但平均值仍然小；
        # - consecutive_frames: 连续满足多少帧后才触发，避免偶然经过导致误切。
        # 这些数值建议先保守测试，再根据日志调节。
        self.auto_q_switch_mean_abs_threshold = 0.22
        self.auto_q_switch_max_abs_threshold = 0.55
        self.auto_q_switch_consecutive_frames = 3
        self.auto_q_switch_min_shooting_steps = 20
        self.auto_q_switch_recovery_hold_steps = 10
        self.auto_q_switch_recovery_ready_steps = 2

        self._auto_q_switch_match_counter = 0
        self._auto_q_switch_has_triggered = False
        self._auto_q_switch_phase = "idle"
        self._auto_q_switch_phase_counter = 0
        self._auto_q_switch_debug_every_n = 20
        self._auto_q_switch_debug_counter = 0

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

    def _spin_wbt_ros_once(self, max_callbacks: int = 4) -> None:
        """非阻塞、线程安全地处理一次 WBT 相关 ROS2 订阅回调。

        中文说明：
        rclpy 的 executor 不能被两个线程同时 spin。键盘线程按 s/c 时会启动
        policy 并捕获 yaw，控制循环也会持续读取 object/waist/torso 话题；
        因此这里统一走同一个 executor + lock，避免 generator already executing。
        """
        executor = getattr(self, "_wbt_executor", None)
        if executor is None:
            return

        lock = getattr(self, "_rclpy_spin_lock", None)
        if lock is None:
            for _ in range(max(1, int(max_callbacks))):
                executor.spin_once(timeout_sec=0.0)
            return

        with lock:
            for _ in range(max(1, int(max_callbacks))):
                executor.spin_once(timeout_sec=0.0)

    def _spin_mocap_subscribers(self) -> None:
        """非阻塞读取一次动捕话题。"""
        self._spin_wbt_ros_once(max_callbacks=4)

    def _get_torso_pose_from_waist_mocap(
        self,
        spin_once: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """由 /waist/pos 的动捕 6D 位姿估计 torso_link 的世界系位姿。

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
        # 其中 p_torso_in_waist = [-0.0039635, 0, 0.034] m。
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
        # 实机动捕模式：优先用 /waist/pos 估计 torso_link 的完整 6D 世界位姿。
        # 这样 get_current_obs_buffer_dict() 里后续的 obj_pos_b 计算，
        # 仍然可以完全沿用 wbt_sim 的四元数转换逻辑。
        mocap_torso_pose = self._get_torso_pose_from_waist_mocap(spin_once=True)
        if mocap_torso_pose is not None:
            return mocap_torso_pose

        # 仿真/兜底模式：保持 wbt_sim 的原始逻辑。
        # 优先使用 MuJoCo 通过 ROS2 发布的 torso_link 世界位置，避免 low_state 中根平移恒为 0。
        self._spin_wbt_ros_once(max_callbacks=2)
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

        # 中文说明：不同导出的 ONNX 输出名可能略有差别。
        # 这里按可用输出名读取 reference；如果没有 ref_pos_xyz，也不会报错。
        output_name_set = set(self.onnx_output_names)
        init_output_names = [
            name
            for name in ["joint_pos", "joint_vel", "ref_pos_xyz", "ref_quat_xyzw"]
            if name in output_name_set
        ]
        init_outputs = self.onnx_policy_session.run(init_output_names, input_feed)
        init_output_map = dict(zip(init_output_names, init_outputs))

        if "joint_pos" not in init_output_map or "joint_vel" not in init_output_map or "ref_quat_xyzw" not in init_output_map:
            raise ValueError(
                "WBT ONNX must provide at least joint_pos, joint_vel and ref_quat_xyzw outputs. "
                f"Available outputs: {self.onnx_output_names}"
            )

        ref_joint_pos = init_output_map["joint_pos"]
        ref_joint_vel = init_output_map["joint_vel"]
        ref_pos_xyz = init_output_map.get("ref_pos_xyz", np.zeros((1, 3), dtype=np.float32))
        ref_quat_xyzw = init_output_map["ref_quat_xyzw"]

        # motion_command_t/ref_quat_xyzw_t will be used in get_current_obs_buffer_dict
        self.motion_command_t = np.concatenate([ref_joint_pos, ref_joint_vel], axis=1)  # (1, 58)
        self.ref_pos_xyz_t = ref_pos_xyz.copy()
        self.ref_quat_xyzw_t = ref_quat_xyzw.copy()

        # duplicate, will be used in _get_init_target and _handle_stop_policy
        self.motion_command_0 = self.motion_command_t.copy()
        self.ref_pos_xyz_0 = self.ref_pos_xyz_t.copy()
        self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()

        # 缓存当前 reference，供按 s/c 时打印。
        self.current_ref_joint_pos = ref_joint_pos.copy()
        self.current_ref_joint_vel = ref_joint_vel.copy()
        self.current_ref_pos_xyz = ref_pos_xyz.copy()
        self.current_ref_quat_xyzw = ref_quat_xyzw.copy()

        def policy_act(input_feed):
            run_output_names = [
                name
                for name in ["actions", "joint_pos", "joint_vel", "ref_pos_xyz", "ref_quat_xyzw"]
                if name in self.onnx_output_names
            ]
            output = self.onnx_policy_session.run(run_output_names, input_feed)
            output_map = dict(zip(run_output_names, output))

            action = output_map["actions"]
            ref_joint_pos = output_map["joint_pos"]
            ref_joint_vel = output_map["joint_vel"]
            ref_pos_xyz = output_map.get("ref_pos_xyz", np.zeros((1, 3), dtype=np.float32))
            ref_quat_xyzw = output_map["ref_quat_xyzw"]

            # 中文说明：实时缓存 ONNX 内部 reference，方便在按键切换时打印调试。
            self.current_ref_joint_pos = ref_joint_pos.copy()
            self.current_ref_joint_vel = ref_joint_vel.copy()
            self.current_ref_pos_xyz = ref_pos_xyz.copy()
            self.current_ref_quat_xyzw = ref_quat_xyzw.copy()

            motion_command = np.concatenate([ref_joint_pos, ref_joint_vel], axis=1)
            return action, motion_command, ref_quat_xyzw

        self.policy = policy_act

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "motion_command_0": self.motion_command_0.copy(),
                "ref_pos_xyz_0": None if self.ref_pos_xyz_0 is None else self.ref_pos_xyz_0.copy(),
                "ref_quat_xyzw_0": self.ref_quat_xyzw_0.copy(),
                "current_ref_joint_pos": None if self.current_ref_joint_pos is None else self.current_ref_joint_pos.copy(),
                "current_ref_joint_vel": None if self.current_ref_joint_vel is None else self.current_ref_joint_vel.copy(),
                "current_ref_pos_xyz": None if self.current_ref_pos_xyz is None else self.current_ref_pos_xyz.copy(),
                "current_ref_quat_xyzw": None if self.current_ref_quat_xyzw is None else self.current_ref_quat_xyzw.copy(),
                # 中文说明：不同 ONNX 里可能携带不同 metadata/URDF；
                # 即便目前两个 policy 用同一个机器人，也保存它可以避免切换后状态不一致。
                "pinocchio_robot": self.pinocchio_robot,
            }
        )
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self.motion_command_0 = state["motion_command_0"].copy()
        self.ref_pos_xyz_0 = None if state.get("ref_pos_xyz_0") is None else state["ref_pos_xyz_0"].copy()
        self.ref_quat_xyzw_0 = state["ref_quat_xyzw_0"].copy()
        self.current_ref_joint_pos = None if state.get("current_ref_joint_pos") is None else state["current_ref_joint_pos"].copy()
        self.current_ref_joint_vel = None if state.get("current_ref_joint_vel") is None else state["current_ref_joint_vel"].copy()
        self.current_ref_pos_xyz = None if state.get("current_ref_pos_xyz") is None else state["current_ref_pos_xyz"].copy()
        self.current_ref_quat_xyzw = None if state.get("current_ref_quat_xyzw") is None else state["current_ref_quat_xyzw"].copy()
        if "pinocchio_robot" in state:
            self.pinocchio_robot = state["pinocchio_robot"]
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_pos_xyz_t = None if self.ref_pos_xyz_0 is None else self.ref_pos_xyz_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        # 重新打印当前 policy 使用的 action offset，方便确认切换生效。
        self._last_logged_policy_default_key = None

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


    def _get_active_policy_default_q(self) -> np.ndarray:
        """Return the deployment-side action offset/default_q for the active policy.

        中文说明：
        WBT ONNX 输出的是 q_action，不是绝对 q_target。训练时每个 policy 使用的
        default_q 不同，因此部署时必须按 active_policy_index 切换 action offset：
        - policy 1 / index 0: shooting reference 第 0 帧；
        - policy 2 / index 1: recovery reference 第 0 帧。
        """
        active_index = int(getattr(self, "active_policy_index", 0))
        if active_index == 1 and hasattr(self, "_recovery_ref0_default_q"):
            return self._recovery_ref0_default_q.astype(np.float32, copy=False).reshape(1, -1)
        if active_index == 0 and hasattr(self, "_shooting_ref0_default_q"):
            return self._shooting_ref0_default_q.astype(np.float32, copy=False).reshape(1, -1)
        return np.asarray(self.default_dof_angles, dtype=np.float32).reshape(1, -1)

    def _log_active_policy_default_q_once(self) -> None:
        """Print the action offset once after policy/default switch for sanity check."""
        key = (
            int(getattr(self, "active_policy_index", -1)),
            bool(getattr(self, "force_zero_obj_pos_b", False)),
            str(getattr(self, "active_model_path", "")),
        )
        if getattr(self, "_last_logged_policy_default_key", None) == key:
            return
        self._last_logged_policy_default_key = key
        default_q = self._get_active_policy_default_q().reshape(-1)
        self.logger.info(
            colored(
                "[POLICY DEFAULT_Q DEBUG] "
                f"active_policy_index={getattr(self, 'active_policy_index', None) + 1 if hasattr(self, 'active_policy_index') else None} | "
                f"force_zero_obj_pos_b={getattr(self, 'force_zero_obj_pos_b', False)} | "
                f"default_q_first6={np.round(default_q[:6], 4).tolist()} | "
                f"default_q_mean={float(np.mean(default_q)):.6f}",
                "cyan",
            )
        )

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
        #print("robot_ref_ori",robot_ref_ori)
        motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
        #print("motion_ref_ori_b",motion_ref_ori_b)
        current_obs_buffer_dict["motion_ref_ori_b"] = motion_ref_ori_b[..., :2].reshape(1, -1)

        # base_ang_vel
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6
        ]

        # dof_pos
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self._get_active_policy_default_q()

        # dof_vel
        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        # actions
        current_obs_buffer_dict["actions"] = self.last_policy_action

        # goal_pos_w 是物体在世界坐标系下的目标位置（来自 motion 文件），
        # 在 obs 中固定不变，与 obj_pos_b 是否被置零无关。
        current_obs_buffer_dict["goal_pos_w"] = np.array([[0.0238, -1.0884, 2.0000]], dtype=np.float32)

        # 中文说明：
        # recovery policy 的训练 obs 中 obj_pos_b 最后三维固定为 0。
        # 一旦按 c 进入恢复阶段，就不要再读取篮球/物体位置，而是直接给零向量。
        # 这样既保持 obs 维度一致，也避免投出后的随机球轨迹干扰恢复 policy。
        if getattr(self, "force_zero_obj_pos_b", False):
            current_obs_buffer_dict["obj_pos_b"] = np.zeros((1, 3), dtype=np.float32)
            if not getattr(self, "_zero_obj_pos_b_debug_printed", False):
                self.logger.info(
                    colored(
                        "[OBJ_POS_B DEBUG] force_zero_obj_pos_b=True, "
                        "set obj_pos_b=[0.0, 0.0, 0.0] for recovery policy.",
                        "cyan",
                    )
                )
                self._zero_obj_pos_b_debug_printed = True
            return current_obs_buffer_dict

        # 有关物体的
        # 实机动捕模式：读取 /basketball/pose 与 /waist/pos，
        # 先由 waist_yaw_link 估计 torso_link，再完全复用 wbt_sim 的 q10 + quat_apply 逻辑计算 obj_pos_b。
        self._spin_mocap_subscribers()
        use_mocap_object = (
            getattr(self.basketball_pose_sub, "received", False)
            and getattr(self.waist_pose_sub, "received", False)
        )

        if use_mocap_object:
            obj_source = "mocap"
            obj_pos_w = self.basketball_pose_sub.latest_pos_w.astype(np.float32).reshape(1, 3)
            torso_pose = self._get_torso_pose_from_waist_mocap(spin_once=False)
            if torso_pose is None:
                raise RuntimeError("waist mocap pose disappeared unexpectedly after received=True.")
            robot_ref_pos_w, robot_ref_quat_wxyz = torso_pose
        else:
            obj_source = "mujoco"
            if not self._mocap_wait_warned:
                self.logger.warning(
                    "Mocap /basketball/pose and /waist/pos are not both received yet; "
                    "falling back to wbt_sim /object_state + /torso_link_pos logic."
                )
                self._mocap_wait_warned = True

            # 仿真/兜底模式：保持 wbt_sim 的原始逻辑。
            self._spin_wbt_ros_once(max_callbacks=2)
            obj_pos_w = self.object_state_sub.latest_pos_w.astype(np.float32).reshape(1, 3)
            robot_ref_pos_w, robot_ref_quat_wxyz = self._get_ref_body_pose_in_world(robot_state_data)

        #print("obj_pos_w",obj_pos_w)
        robot_ref_pos_w = robot_ref_pos_w.astype(np.float32).reshape(1, 3)
        robot_ref_quat_xyzw = wxyz_to_xyzw(robot_ref_quat_wxyz).astype(np.float32).reshape(1, 4)

        # 完全同构 wbt_sim / 训练里的 subtract_frame_transforms 位置部分：
        # q10 = quat_inverse(q01, w_last=True)
        # t12 = quat_apply(q10, t02 - t01, w_last=True)
        q10_xyzw = self._quat_conjugate_xyzw(robot_ref_quat_xyzw)
        obj_pos_b = self._quat_apply_xyzw(q10_xyzw, obj_pos_w - robot_ref_pos_w).astype(np.float32)
        #obj_pos_b[:, 2] += 1.0

        current_obs_buffer_dict["obj_pos_b"] = obj_pos_b.reshape(1, 3)

        if getattr(self, "debug_obj_pos_b_source_enabled", False):
            self._debug_obj_pos_b_counter += 1
            every_n = max(int(getattr(self, "debug_obj_pos_b_every_n", 50)), 1)
            source_changed = obj_source != getattr(self, "_last_obj_pos_b_source", None)

            if source_changed or self._debug_obj_pos_b_counter % every_n == 0:
                # self.logger.info(
                #     "[OBJ_POS_B DEBUG] "
                #     f"source={obj_source} | "
                #     f"use_mocap_object={use_mocap_object} | "
                #     f"basketball_received={getattr(self.basketball_pose_sub, 'received', False)} | "
                #     f"waist_received={getattr(self.waist_pose_sub, 'received', False)} | "
                #     f"obj_pos_b={np.round(obj_pos_b.reshape(-1), 4).tolist()} | "
                #     f"obj_pos_w={np.round(obj_pos_w.reshape(-1), 4).tolist()} | "
                #     f"ref_pos_w={np.round(robot_ref_pos_w.reshape(-1), 4).tolist()}"
                # )
                self._last_obj_pos_b_source = obj_source

        #print('current_obs_buffer_dict["obj_pos_b"]',current_obs_buffer_dict["obj_pos_b"])

        #obj_ori_w_mat = obj_ori_w.reshape(3, 2).astype(np.float32)
        #obj_ori_b_mat = (robot_ref_rot.T @ obj_ori_w_mat).astype(np.float32)
        #current_obs_buffer_dict["obj_ori_b"] = obj_ori_b_mat.reshape(1, 6)

        #obj_lin_vel_b = quat_rotate_inverse(robot_ref_quat_wxyz, obj_lin_vel_w).astype(np.float32)
        #current_obs_buffer_dict["obj_lin_vel_b"] = obj_lin_vel_b.reshape(1, 3)

        return current_obs_buffer_dict

    def rl_inference(self, robot_state_data):
        # prepare obs, run policy inference
        if not self.motion_clip_progressing:
            # Keep motion index pinned at the start while waiting to trigger the clip.
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None

        self._print_recovery_first_real_joint_pos_if_needed(robot_state_data)

        obs = self.prepare_obs_for_rl(robot_state_data)
        #obs["actor_obs"][0,-1]+=0.5
        #print('obs["actor_obs"]:',obs["actor_obs"][0,-3:])
        #print("obs.shape:",len(obs))
        input_feed = {"time_step": np.array([[self.motion_timestep]], dtype=np.float32), "obs": obs["actor_obs"]}
        policy_action, self.motion_command_t, self.ref_quat_xyzw_t = self.policy(input_feed)
        self.ref_pos_xyz_t = None if self.current_ref_pos_xyz is None else self.current_ref_pos_xyz.copy()

        # clip policy action
        policy_action = np.clip(policy_action, -100, 100)
        # store last policy action
        self.last_policy_action = policy_action.copy()
        # scale policy action
        self.scaled_policy_action = policy_action * self.policy_action_scale
        # 中文说明：推理侧 action_scale / kp / kd / p_gain / d_gain 的统一日志
        # 在 BasePolicy.policy_action() 中打印，因为那里已经拿到了 q_target，
        # 更方便和 IsaacSim joint_control 的 action_scales / p_gains / d_gains 对齐。
        # print("policy_action", policy_action)
        # print("wbt.self.scaled_policy_action", self.scaled_policy_action)

        # update motion timestep
        if self.motion_clip_progressing:
            if self.use_sim_time:
                self._update_clock()
            else:
                self.motion_timestep += 1

        # 自动切换检测：
        # 不是用固定 frame 触发，而是用当前机器人 q 与 reference 第224帧 q 的相似度触发。
        self._auto_q_similarity_switch_update(robot_state_data)

        return self.scaled_policy_action

    def policy_action(self):
        """Execute WBT policy action using per-policy default_q/action offset.

        中文说明：
        BasePolicy 默认使用 q_target = scaled_policy_action + self.default_dof_angles。
        但当前双 policy 中，shooting / recovery 的训练 default_q 不同，所以这里重写
        policy_action，在真正发送 q_target 时按 active_policy_index 选择对应的 ref0 q。
        """
        kp_override = None
        kd_override = None
        scaled_policy_action = None

        # Stage 1: Read State
        with self.latency_tracker.measure("read_state"):
            robot_state_data = self.interface.get_low_state()

        # 自动 q 相似度触发后会在 rl_inference() 中直接切到 recovery tracking，
        # 这里不再执行 recovery_ref0 hold / ready 的分阶段推进。

        # Stage 2: Pre-processing
        with self.latency_tracker.measure("preprocessing"):
            if self.get_ready_state:
                q_target = self.get_init_target(robot_state_data)
                self.init_count = min(self.init_count, 500)
            elif not self.use_policy_action:
                manual_cmd = self._get_manual_command(robot_state_data)
                if manual_cmd is not None:
                    q_target = manual_cmd["q"]
                    kp_override = manual_cmd.get("kp")
                    kd_override = manual_cmd.get("kd")
                else:
                    q_target = robot_state_data[:, 7 : 7 + self.num_dofs]

        # Stage 3: Inference
        if self.use_policy_action and not self.get_ready_state:
            with self.latency_tracker.measure("inference"):
                scaled_policy_action = self.rl_inference(robot_state_data)

        # Stage 4: Post-processing
        with self.latency_tracker.measure("postprocessing"):
            if self.use_policy_action and not self.get_ready_state:
                if scaled_policy_action.shape[1] != self.num_dofs:
                    if not self.upper_body_controller:
                        scaled_policy_action = np.concatenate(
                            [np.zeros((1, self.num_dofs - scaled_policy_action.shape[1])), scaled_policy_action],
                            axis=1,
                        )
                    else:
                        raise NotImplementedError("Upper body controller not implemented")

                active_default_q = self._get_active_policy_default_q()
                q_target = scaled_policy_action + active_default_q
                self._log_active_policy_default_q_once()

            self.cmd_q[:] = q_target[0]

            if self.use_policy_action and not self.get_ready_state:
                self._debug_print_inference_control_params(
                    robot_state_data=robot_state_data,
                    q_target=q_target,
                    scaled_policy_action=scaled_policy_action,
                    kp_override=kp_override,
                    kd_override=kd_override,
                )

        # Stage 5: Action Pub
        with self.latency_tracker.measure("action_pub"):
            self.interface.send_low_command(
                self.cmd_q,
                self.cmd_dq,
                self.cmd_tau,
                robot_state_data[0, 7 : 7 + self.num_dofs],
                kp_override=kp_override,
                kd_override=kd_override,
            )

    def _get_manual_command(self, robot_state_data):
        # TODO: instead of adding kp/kd_override in def _set_motor_command,
        # just use the motor_kp/motor_kd when calling it in _fill_motor_commands
        if not self._stiff_hold_active:
            return None

        # 中文说明：manual hold 统一使用 stiff_startup_kp / stiff_startup_kd。
        # - stiff_startup：按 ] 后保持原始 stiff_startup_pos；
        # - recovery_ref0：按 c 后固定到恢复 policy 第 0 帧 reference joint_pos。
        hold_target = getattr(self, "_manual_hold_target", "stiff_startup")
        if hold_target == "recovery_ref0":
            q_target = self._recovery_ref0_hold_q
        else:
            q_target = self._stiff_hold_q

        return {
            "q": q_target.copy(),
            "kp": self._stiff_hold_kp,
            "kd": self._stiff_hold_kd,
        }

    def _enable_shoot_policy_without_motion(self):
        """按 ] 后启用投篮 policy，但不播放 motion clip。

        中文说明：
        - 这一步恢复为原来的 WBT 使用习惯：] 只是进入 policy 控制；
        - motion_timestep 固定在 0，因此投篮 policy 会停在第 0 帧附近；
        - 再按 s 才真正从第 0 帧开始播放 shooting tracking。
        """
        if len(self.model_paths) < 1:
            self.logger.warning(colored("Cannot enable shooting policy: no policy is loaded.", "yellow"))
            return

        if self.active_policy_index != 0:
            self._activate_policy(0)

        self._reset_auto_q_switch_state()

        self.force_zero_obj_pos_b = False
        self._zero_obj_pos_b_debug_printed = False

        self._handle_start_policy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._print_current_reference_debug("shooting_ready")
        self.logger.info(colored("Enabled shooting policy at frame 0. Press s to start shooting tracking.", "blue"))

    def _hold_recovery_ref0_pose(self):
        """按 c 后固定到恢复 policy 第 0 帧 reference joint_pos。"""
        if len(self.model_paths) < 2:
            self.logger.warning(
                colored(
                    f"Cannot hold recovery ref0: recovery policy slot [2] is not available. "
                    f"Loaded policy count = {len(self.model_paths)}.",
                    "yellow",
                )
            )
            return

        if self.active_policy_index != 1:
            self._activate_policy(1)

        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self._manual_hold_target = "recovery_ref0"
        self.force_zero_obj_pos_b = True
        self._zero_obj_pos_b_debug_printed = False
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

        self._print_current_reference_debug("recovery_ref0_hold")
        self.logger.info(
            colored(
                "Holding recovery policy frame-0 ref_joint_pos with stiff startup KP/KD. "
                "Press x to start recovery tracking.",
                "yellow",
            )
        )

    def _handle_start_policy(self):
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._manual_hold_target = "none"
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
        self._manual_hold_target = "stiff_startup"
        self.force_zero_obj_pos_b = False
        self._zero_obj_pos_b_debug_printed = False

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.clock_sub.reset_origin()
        self.motion_clip_progressing = True
        # Capture motion-specific start timestep for policy-level timing control
        self.motion_start_timestep = None  # will be set in rl_inference
        self.motion_timestep = 0  # Reset to start from beginning of motion
        self._last_clock_reading = None
        self.logger.info(colored("Starting motion clip", "blue"))

    def _format_np_array_for_log(self, value, precision: int = 4, max_items: int | None = None):
        """把 numpy 数组整理成适合日志打印的一维列表。"""
        if value is None:
            return None
        arr = np.asarray(value).reshape(-1)
        if max_items is not None:
            arr = arr[:max_items]
        return np.round(arr.astype(np.float64), precision).tolist()

    def _refresh_reference_debug_cache_at_current_timestep(self) -> None:
        """在当前 motion_timestep 下主动跑一次 ONNX，刷新当前 reference 缓存。"""
        actor_obs_template = self.obs_buf_dict.get("actor_obs")
        if actor_obs_template is None:
            return

        input_feed = {
            "time_step": np.array([[self.motion_timestep]], dtype=np.float32),
            "obs": actor_obs_template.copy(),
        }
        run_output_names = [
            name
            for name in ["joint_pos", "joint_vel", "ref_pos_xyz", "ref_quat_xyzw"]
            if name in self.onnx_output_names
        ]
        if not run_output_names:
            return

        try:
            outputs = self.onnx_policy_session.run(run_output_names, input_feed)
        except Exception as exc:
            self.logger.warning(f"Failed to refresh ONNX reference cache: {exc}")
            return

        output_map = dict(zip(run_output_names, outputs))
        if "joint_pos" in output_map:
            self.current_ref_joint_pos = output_map["joint_pos"].copy()
        if "joint_vel" in output_map:
            self.current_ref_joint_vel = output_map["joint_vel"].copy()
        if "ref_pos_xyz" in output_map:
            self.current_ref_pos_xyz = output_map["ref_pos_xyz"].copy()
        if "ref_quat_xyzw" in output_map:
            self.current_ref_quat_xyzw = output_map["ref_quat_xyzw"].copy()

    def _print_current_reference_debug(self, stage_name: str) -> None:
        """按 s/c 启动阶段时打印当前 ONNX 内置 reference 信息。"""
        if not getattr(self, "debug_reference_on_stage_start_enabled", True):
            return

        self._refresh_reference_debug_cache_at_current_timestep()

        joint_preview_count = int(getattr(self, "debug_reference_joint_preview_count", 6))
        ref_joint_pos = getattr(self, "current_ref_joint_pos", None)
        ref_joint_vel = getattr(self, "current_ref_joint_vel", None)
        ref_pos_xyz = getattr(self, "current_ref_pos_xyz", None)
        ref_quat_xyzw = getattr(self, "current_ref_quat_xyzw", None)

        msg_lines = [
            "=" * 88,
            f"[WBT REFERENCE DEBUG] stage={stage_name}",
            f"policy_index={getattr(self, 'active_policy_index', None) + 1 if hasattr(self, 'active_policy_index') else None}",
            f"policy_path={getattr(self, 'active_model_path', None)}",
            f"motion_timestep={self.motion_timestep}",
            f"onnx_outputs={list(getattr(self, 'onnx_output_names', []))}",
            f"ref_pos_xyz={self._format_np_array_for_log(ref_pos_xyz)}",
            f"ref_quat_xyzw={self._format_np_array_for_log(ref_quat_xyzw)}",
            f"ref_joint_pos_preview={self._format_np_array_for_log(ref_joint_pos, max_items=joint_preview_count)}",
            f"ref_joint_vel_preview={self._format_np_array_for_log(ref_joint_vel, max_items=joint_preview_count)}",
        ]

        if ref_joint_pos is not None:
            msg_lines.append(f"ref_joint_pos_mean={float(np.mean(ref_joint_pos)):.6f}")
        if ref_joint_vel is not None:
            msg_lines.append(f"ref_joint_vel_mean={float(np.mean(ref_joint_vel)):.6f}")

        msg_lines.append("=" * 88)
        self.logger.info(colored("\n" + "\n".join(msg_lines), "cyan"))

    def _print_recovery_first_real_joint_pos_if_needed(self, robot_state_data) -> None:
        """按 x 启动 recovery tracking 后，打印第一帧读到的真实 29 自由度关节角。

        中文说明：
        - robot_state_data[:, 7:7+num_dofs] 是部署端读到的真实/仿真机器人当前关节位置；
        - 这里只打印一次，用来检查从 recovery_ref0 固定姿态切入 recovery tracking 时，
          机器人真实姿态是否已经接近 recovery policy 的第 0 帧 reference。
        """
        if not getattr(self, "_print_recovery_first_real_joint_pos_pending", False):
            return

        try:
            real_joint_pos = np.asarray(
                robot_state_data[:, 7 : 7 + self.num_dofs],
                dtype=np.float64,
            ).reshape(-1)
        except Exception as exc:
            self.logger.warning(f"Failed to read first real joint_pos after x: {exc}")
            self._print_recovery_first_real_joint_pos_pending = False
            return

        self._print_recovery_first_real_joint_pos_pending = False

        msg_lines = [
            "=" * 88,
            "[WBT RECOVERY FIRST REAL JOINT_POS AFTER X]",
            f"policy_index={getattr(self, 'active_policy_index', None) + 1 if hasattr(self, 'active_policy_index') else None}",
            f"policy_path={getattr(self, 'active_model_path', None)}",
            f"motion_timestep={self.motion_timestep}",
            f"num_dofs={self.num_dofs}",
            "real_joint_pos_29=[",
        ]
        for value in real_joint_pos[: self.num_dofs]:
            msg_lines.append(f"    {float(value): .8f},")
        msg_lines.extend([
            "]",
            f"real_joint_pos_mean={float(np.mean(real_joint_pos[: self.num_dofs])):.6f}",
            "=" * 88,
        ])
        self.logger.info(colored("\n" + "\n".join(msg_lines), "magenta"))

    def _reset_obs_history_to_zeros(self) -> None:
        """Reset observation history buffers to zeros after switching stages.

        中文说明：
        从 shooting 切到 recovery 时，如果 observation 有 history，上一阶段的
        actions / motion_command / obj_pos_b 可能残留在历史帧里。这里统一清零，
        避免 recovery policy 在刚启用时吃到 shooting 阶段的历史观测。
        """
        try:
            for group, term_buffers in getattr(self, "obs_history_buffers", {}).items():
                history_len = self.history_length_dict.get(group, 1)
                for term, buffer in term_buffers.items():
                    buffer.clear()
                    term_dim = self.obs_dims[term]
                    zero_obs = np.zeros((1, term_dim), dtype=np.float32)
                    for _ in range(history_len):
                        buffer.append(zero_obs.copy())

            for group, term_names in getattr(self, "obs_terms_sorted", {}).items():
                history_len = self.history_length_dict.get(group, 1)
                flattened_terms = []
                for term in term_names:
                    term_dim = self.obs_dims[term]
                    flattened_terms.append(np.zeros((1, term_dim * history_len), dtype=np.float32))
                self.obs_buf_dict[group] = (
                    np.concatenate(flattened_terms, axis=1).astype(np.float32, copy=False)
                    if flattened_terms
                    else np.zeros((1, 0), dtype=np.float32)
                )

            self.logger.info("Observation history buffers reset to zero.")
        except Exception as exc:
            self.logger.warning(f"Failed to reset observation history buffers: {exc}")

    def _enable_recovery_policy_without_motion(self):
        """按 x 后启用 recovery policy，但不播放 recovery motion clip。

        中文说明：
        - c：只是固定到 recovery 第 0 帧 reference joint_pos；
        - x：启用 recovery policy，并让 motion_timestep 固定在 0；
        - v：才真正从第 0 帧开始播放 recovery tracking。
        """
        if len(self.model_paths) < 2:
            self.logger.warning(
                colored(
                    f"Cannot enable recovery policy: recovery policy slot [2] is not available. "
                    f"Loaded policy count = {len(self.model_paths)}.",
                    "yellow",
                )
            )
            return

        if self.active_policy_index != 1:
            self._activate_policy(1)

        # recovery policy 训练时 obj_pos_b 恒为 0，因此 recovery 准备/启用/tracking 全程置零。
        self.force_zero_obj_pos_b = True
        self._zero_obj_pos_b_debug_printed = False

        # 切换 skill 后清空历史观测，避免 shooting 的 actions/motion_command 残留。
        self._reset_obs_history_to_zeros()

        # 与 ] 启用 shooting policy 类似：启用 policy action，但不推进 motion clip。
        self._handle_start_policy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None

        self._print_current_reference_debug("recovery_ready")
        self.logger.info(
            colored(
                "Enabled recovery policy at frame 0 with obj_pos_b=0. "
                "Press v to start recovery tracking.",
                "blue",
            )
        )



    def _auto_q_similarity_switch_update(self, robot_state_data) -> None:
        """Use current robot q similarity to shooting reference frame 224 to trigger recovery.

        中文说明：
        该逻辑不使用球、不使用固定 motion_timestep 作为切换条件；
        只判断当前机器人 29 维 q 是否足够接近指定的 shooting 完成帧 q_ref。
        """
        if not getattr(self, "auto_q_switch_enabled", False):
            return
        if getattr(self, "_auto_q_switch_has_triggered", False):
            return

        # 只在 shooting policy 正在 tracking 时检测。
        if int(getattr(self, "active_policy_index", 0)) != 0:
            return
        if not getattr(self, "motion_clip_progressing", False):
            self._auto_q_switch_match_counter = 0
            return
        if not getattr(self, "use_policy_action", False):
            self._auto_q_switch_match_counter = 0
            return
        if int(getattr(self, "motion_timestep", 0)) < int(getattr(self, "auto_q_switch_min_shooting_steps", 0)):
            return

        try:
            q_cur = np.asarray(
                robot_state_data[:, 7 : 7 + self.num_dofs],
                dtype=np.float32,
            ).reshape(1, -1)
            q_ref = np.asarray(self.auto_q_switch_q_ref, dtype=np.float32).reshape(1, -1)
        except Exception as exc:
            self.logger.warning(f"[AUTO Q SWITCH] Failed to read q for similarity check: {exc}")
            return

        abs_err = np.abs(q_cur - q_ref)
        mean_abs_err = float(np.mean(abs_err))
        max_abs_err = float(np.max(abs_err))

        matched = (
            mean_abs_err <= float(getattr(self, "auto_q_switch_mean_abs_threshold", 0.18))
            and max_abs_err <= float(getattr(self, "auto_q_switch_max_abs_threshold", 0.55))
        )

        if matched:
            self._auto_q_switch_match_counter += 1
        else:
            self._auto_q_switch_match_counter = 0

        self._auto_q_switch_debug_counter += 1
        every_n = max(int(getattr(self, "_auto_q_switch_debug_every_n", 20)), 1)
        # if matched or self._auto_q_switch_debug_counter % every_n == 0:
        #     self.logger.info(
        #         "[AUTO Q SWITCH DEBUG] "
        #         #f"frame_ref={getattr(self, 'auto_q_switch_target_frame', None)} | "
        #         f"motion_timestep={getattr(self, 'motion_timestep', None)} | "
        #         f"mean_abs_err={mean_abs_err:.4f} | "
        #         f"max_abs_err={max_abs_err:.4f} | "
        #         f"matched={matched} | "
        #         f"counter={self._auto_q_switch_match_counter}/"
        #         f"{getattr(self, 'auto_q_switch_consecutive_frames', None)}"
        #     )

        if self._auto_q_switch_match_counter >= int(getattr(self, "auto_q_switch_consecutive_frames", 3)):
            self._auto_q_switch_has_triggered = True
            self._auto_q_switch_phase = "done"
            self._auto_q_switch_phase_counter = 0
            self.logger.info(
                colored(
                    "[AUTO Q SWITCH] Shooting completion pose matched. "
                    "Switching directly to recovery policy tracking automatically.",
                    "green",
                )
            )

            # 中文说明：
            # 用户要求不要再插入 recovery_ref0 stiff hold。命中投掷完成姿态后，
            # 直接切到 recovery policy 并从第 0 帧开始 tracking。
            # 切换前清空 observation history，避免 shooting 阶段的 actions / motion_command
            # 残留到 recovery policy 的历史观测中。
            self._reset_obs_history_to_zeros()
            self._start_policy_slot_and_motion_clip(1, "recovery_auto")

    def _auto_q_switch_sequence_update(self, robot_state_data) -> None:
        """No-op placeholder for the old staged auto-switch sequence.

        中文说明：
        之前自动切换是 shooting -> recovery_ref0 hold -> recovery ready -> recovery tracking。
        当前版本按用户要求改为：shooting 命中 q 相似条件后，直接切到 recovery tracking，
        所以这里保留空函数以兼容已有调用/后续扩展。
        """
        return

    def _reset_auto_q_switch_state(self) -> None:
        """Reset automatic q-similarity switch state.

        中文说明：
        手动重新进入 shooting 时重置自动切换状态，允许下一次投篮重新触发。
        """
        self._auto_q_switch_match_counter = 0
        self._auto_q_switch_has_triggered = False
        self._auto_q_switch_phase = "idle"
        self._auto_q_switch_phase_counter = 0
        self._auto_q_switch_debug_counter = 0


    def _start_policy_slot_and_motion_clip(self, policy_index: int, stage_name: str):
        """Switch to a policy slot and immediately start its WBT motion clip.

        中文说明：
        - policy_index=0 默认是投篮 policy；
        - policy_index=1 默认是恢复 policy；
        - 该函数会重新捕获 robot/motion yaw offset，并从该 policy 的第 0 帧开始 tracking。
        """
        if not (0 <= policy_index < len(self.model_paths)):
            self.logger.warning(
                colored(
                    f"Cannot start {stage_name}: policy slot {policy_index + 1} is not available. "
                    f"Loaded policy count = {len(self.model_paths)}.",
                    "yellow",
                )
            )
            return

        if policy_index != self.active_policy_index:
            self._activate_policy(policy_index)

        if policy_index == 0:
            self._reset_auto_q_switch_state()

        # 中文说明：
        # policy_index=1 是 recovery policy。恢复 policy 训练时 obj_pos_b 恒为 0，
        # 所以启动 recovery tracking 后也继续强制置零；shooting policy 则恢复真实 obj_pos_b。
        self.force_zero_obj_pos_b = (policy_index == 1)
        #self.force_zero_obj_pos_b = (policy_index == 0)
        self._zero_obj_pos_b_debug_printed = False

        # 与手动按 ] 的逻辑保持一致：进入 policy action，退出 stiff hold，重新捕获 yaw 对齐。
        self._handle_start_policy()
        self._handle_start_motion_clip()
        if policy_index == 1 and stage_name == "recovery":
            # 中文说明：按 x 启动 recovery tracking 后，在下一次 rl_inference 中打印第一帧真实关节角。
            self._print_recovery_first_real_joint_pos_pending = True
        self._print_current_reference_debug(stage_name)
        self.logger.info(colored(f"Started {stage_name} tracking with policy [{policy_index + 1}]", "green"))

    def handle_keyboard_button(self, keycode):
        """Add WBT keyboard shortcuts for two-stage basketball tracking.

        中文说明：
        - ]：启用投篮 policy，但 motion_timestep 固定在 0，不开始 tracking；
        - s：开始投篮 policy tracking；
        - c：切到恢复 policy，并固定到恢复 policy 第 0 帧 ref_joint_pos；
        - x：启用 recovery policy，但 motion_timestep 固定在 0，不开始 tracking；
        - v：真正开始 recovery policy tracking。
        """
        if keycode == "]":
            self._enable_shoot_policy_without_motion()
        elif keycode == "s":
            # s：投篮阶段，默认使用第 1 个 policy 并开始 tracking。
            self._start_policy_slot_and_motion_clip(0, "shooting")
        elif keycode == "c":
            # c：只固定到恢复 policy 的第 0 帧 reference joint_pos，不启动恢复 policy。
            self._hold_recovery_ref0_pose()
        elif keycode == "x":
            # x：启用 recovery policy，但不推进 reference motion。
            self._enable_recovery_policy_without_motion()
        elif keycode == "v":
            # v：从 recovery policy 第 0 帧开始真正 tracking。
            self._start_policy_slot_and_motion_clip(1, "recovery")
        else:
            super().handle_keyboard_button(keycode)

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses for WBT-specific controls."""
        if cur_key == "start":
            # Start playing current policy's motion clip.
            self._handle_start_policy()
            self._handle_start_motion_clip()
        else:
            # Delegate all other buttons to base class. In base.py, select cycles policy slots.
            super().handle_joystick_button(cur_key)
        super()._print_control_status()

    def _capture_robot_yaw_offset(self):
        """Capture robot yaw when policy starts to use as reference offset."""
        robot_state_data = self.interface.get_low_state()
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
            executor = getattr(self, "_wbt_executor", None)
            if executor is not None:
                for _node in (
                    getattr(self, "object_state_sub", None),
                    getattr(self, "torso_link_pos_sub", None),
                    getattr(self, "basketball_pose_sub", None),
                    getattr(self, "waist_pose_sub", None),
                ):
                    if _node is not None:
                        try:
                            executor.remove_node(_node)
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
