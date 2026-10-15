import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pinocchio as pin
from defusedxml import ElementTree
from loguru import logger
from termcolor import colored

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some pure-inference environments
    torch = None

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.math.misc import get_index_of_a_in_b
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_mul,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


class PinocchioRobot:
    def __init__(self, robot_cfg: RobotConfig, urdf_text: str):
        xml_text = self._create_xml_from_urdf(urdf_text)
        self.robot_model = pin.buildModelFromXML(xml_text, pin.JointModelFreeFlyer())
        self.robot_data = self.robot_model.createData()

        joint_names_in_real_robot = robot_cfg.dof_names
        joint_names_in_pinocchio_robot = [
            name for name in self.robot_model.names if name not in ["universe", "root_joint"]
        ]
        assert len(joint_names_in_pinocchio_robot) == len(joint_names_in_real_robot), (
            "The number of joints in the pinocchio robot and the real robot are not the same"
        )
        self.real2pinocchio_index = get_index_of_a_in_b(joint_names_in_pinocchio_robot, joint_names_in_real_robot)

        self.ref_body_frame_id = self.robot_model.getFrameId(robot_cfg.motion["body_name_ref"][0])

    def fk_and_get_ref_body_pose_in_world(self, configuration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pin.framesForwardKinematics(self.robot_model, self.robot_data, configuration)
        ref_body_pose_in_world = self.robot_data.oMf[self.ref_body_frame_id]
        position = np.expand_dims(np.asarray(ref_body_pose_in_world.translation), axis=0)
        quaternion = pin.Quaternion(ref_body_pose_in_world.rotation)
        return position, np.expand_dims(quaternion.coeffs(), axis=0)

    def fk_and_get_ref_body_orientation_in_world(self, configuration: np.ndarray) -> np.ndarray:
        _, quat_xyzw = self.fk_and_get_ref_body_pose_in_world(configuration)
        return quat_xyzw

    @staticmethod
    def _create_xml_from_urdf(urdf_text: str) -> str:
        root = ElementTree.fromstring(urdf_text)

        def _is_visual_or_collision(tag: str) -> bool:
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
        self.latest_quat_wxyz = np.zeros((1, 4), dtype=np.float32)
        self.quat_received = False
        self.received = False
        self.create_subscription(Float32MultiArray, topic_name, self._callback, 10)

    def _callback(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size < 3:
            self.get_logger().warning(f"/torso_link_pos data size is {data.size}, expected at least 3")
            return
        self.latest_pos_w = data[0:3].reshape(1, 3)
        if data.size >= 7:
            self.latest_quat_wxyz = data[3:7].reshape(1, 4)
            self.quat_received = True
        self.received = True


class WholeBodyTrackingPolicy(BasePolicy):
    def __init__(self, config: InferenceConfig):
        self.goal_pos_w = np.array([config.task.goal_pos_w], dtype=np.float32)
        self.motion_timestep = 0
        self.motion_clip_progressing = False
        self.motion_start_timestep = None
        self.motion_command_t = None
        self.ref_quat_xyzw_t = None
        self.motion_command_0 = None
        self.ref_quat_xyzw_0 = None

        self.timestep_interval_ms = 1000.0 / config.task.rl_rate

        self.clock_sub = ClockSub()
        self.clock_sub.start()
        self._last_clock_reading: int | None = None

        self.use_sim_time = config.task.use_sim_time

        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

        if not rclpy.ok():
            rclpy.init(args=None)

        # Subscribe to MuJoCo simulation topics for object and torso positions.
        self.object_state_sub = ObjectStateSub(topic_name="/object_state")
        self.torso_link_pos_sub = TorsoLinkPosSub(topic_name="/torso_link_pos")

        self.debug_obj_pos_b_source_enabled = True
        self.debug_obj_pos_b_every_n = 50
        self._debug_obj_pos_b_counter = 0
        self._last_obj_pos_b_source = None

        super().__init__(config)

        # Load stiff startup parameters from robot config
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
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

        def _show_warning():
            logger.warning(
                colored(
                    "\u26a0\ufe0f  Non-interactive mode detected - cannot prompt for stiff mode confirmation!",
                    "red",
                    attrs=["bold"],
                )
            )

        if getattr(self, "_skip_stiff_hold_prompt", False):
            logger.info("Stiff hold prompt disabled by guarded deployment entrypoint.")
        elif sys.stdin.isatty():
            logger.info(colored("\n\u26a0\ufe0f  Ready to enter stiff hold mode", "yellow", attrs=["bold"]))
            logger.info(colored("Press Enter to continue...", "yellow"))
            try:
                input()
                logger.info(colored("\u2713 Entering stiff hold mode", "green"))
            except EOFError:
                _show_warning()
        else:
            _show_warning()

    @staticmethod
    def _quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
        q = q.copy()
        q[:, 0:3] *= -1.0
        return q

    @staticmethod
    def _quat_apply_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_vec = q[:, 0:3]
        q_w = q[:, 3:4]
        t = 2.0 * np.cross(q_vec, v)
        return v + q_w * t + np.cross(q_vec, t)

    def _get_ref_body_pose_in_world(self, robot_state_data) -> tuple[np.ndarray, np.ndarray]:
        """Get reference body (torso_link) pose in world frame.

        Uses /torso_link_pos from MuJoCo if available, falls back to robot_state_data.
        """
        rclpy.spin_once(self.torso_link_pos_sub, timeout_sec=0.0)
        if self.torso_link_pos_sub.received and self.torso_link_pos_sub.quat_received:
            return (
                self.torso_link_pos_sub.latest_pos_w.astype(np.float32).reshape(1, 3),
                self.torso_link_pos_sub.latest_quat_wxyz.astype(np.float32).reshape(1, 4),
            )
        if self.torso_link_pos_sub.received:
            root_pos = self.torso_link_pos_sub.latest_pos_w[0].astype(np.float32)
        else:
            root_pos = robot_state_data[0, :3].astype(np.float32)

        root_ori_xyzw = wxyz_to_xyzw(robot_state_data[:, 3:7])[0]

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
        """Initialize object reset logic for ']' key.

        Reads the initial object pose relative to the reference body from the npz,
        so pressing ']' places the object at the correct world position relative
        to the robot's current pose.
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

        manual_object_pos_w = motion_cfg.get("object_init_world_pos", None)
        manual_object_quat_wxyz = (
            motion_cfg.get("object_init_world_quat_wxyz", None)
            or motion_cfg.get("object_init_world_quat", None)
        )

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
        if self.object_init_rel_pos_ref is None or self.object_init_rel_quat_ref_wxyz is None:
            raise RuntimeError("Object relative pose is not initialized yet.")

        ref_rot_w = self._quat_to_rotmat_wxyz(current_ref_quat_wxyz)
        target_pos_w = current_ref_pos_w + ref_rot_w @ self.object_init_rel_pos_ref
        target_quat_wxyz = self._quat_mul_wxyz(current_ref_quat_wxyz, self.object_init_rel_quat_ref_wxyz)
        return target_pos_w, target_quat_wxyz

    def _resolve_simulator_handle(self):
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
        pos_w = np.asarray(pos_w, dtype=np.float64).reshape(3)
        quat_wxyz = self._quat_normalize_wxyz(quat_wxyz)

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

    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        time_step = np.zeros((1, 1), dtype=np.float32)

        actor_obs_template = self.obs_buf_dict.get("actor_obs")
        if actor_obs_template is None:
            raise ValueError("Observation group 'actor_obs' must be configured for WBT policy.")
        obs = actor_obs_template.copy()
        input_feed = {"obs": obs, "time_step": time_step}
        outputs = self.onnx_policy_session.run(["joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)

        self.motion_command_t = np.concatenate(outputs[0:2], axis=1)
        self.ref_quat_xyzw_t = outputs[2]
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
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            target_dof_pos = self.motion_command_0[:, : self.num_dofs]
            q_target = dof_pos + (target_dof_pos - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = {}

        current_obs_buffer_dict["motion_command"] = self.motion_command_t

        motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)
        motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)
        robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)
        motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
        current_obs_buffer_dict["motion_ref_ori_b"] = motion_ref_ori_b[..., :2].reshape(1, -1)

        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6
        ]

        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles

        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        current_obs_buffer_dict["actions"] = self.last_policy_action

        current_obs_buffer_dict["goal_pos_w"] = self.goal_pos_w

        # obj_pos_b: read real object position from MuJoCo /object_state,
        # then compute relative to reference body pose.
        rclpy.spin_once(self.object_state_sub, timeout_sec=0.0)
        obj_pos_w = self.object_state_sub.latest_pos_w.astype(np.float32).reshape(1, 3)
        robot_ref_pos_w, robot_ref_quat_wxyz = self._get_ref_body_pose_in_world(robot_state_data)

        robot_ref_pos_w = robot_ref_pos_w.astype(np.float32).reshape(1, 3)
        robot_ref_quat_xyzw = wxyz_to_xyzw(robot_ref_quat_wxyz).astype(np.float32).reshape(1, 4)

        q10_xyzw = self._quat_conjugate_xyzw(robot_ref_quat_xyzw)
        obj_pos_b = self._quat_apply_xyzw(q10_xyzw, obj_pos_w - robot_ref_pos_w).astype(np.float32)

        current_obs_buffer_dict["obj_pos_b"] = obj_pos_b.reshape(1, 3)

        return current_obs_buffer_dict

    def rl_inference(self, robot_state_data):
        if not self.motion_clip_progressing:
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None

        obs = self.prepare_obs_for_rl(robot_state_data)

        input_feed = {"time_step": np.array([[self.motion_timestep]], dtype=np.float32), "obs": obs["actor_obs"]}
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

    def _get_manual_command(self, robot_state_data):
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
        current_clock = self.clock_sub.get_clock()
        if self.motion_start_timestep is None:
            self.motion_start_timestep = current_clock
        elif self._last_clock_reading is not None and current_clock < self._last_clock_reading:
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
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self.logger.info("Actions set to stiff startup command")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_command_t = self.motion_command_0.copy()
        self._last_clock_reading = None
        self.robot_yaw_offset = 0.0

    def _handle_start_motion_clip(self):
        self.clock_sub.reset_origin()
        self.motion_clip_progressing = True
        self.motion_start_timestep = None
        self.motion_timestep = 0
        self._last_clock_reading = None
        self.logger.info(colored("Starting motion clip", "blue"))

    def handle_keyboard_button(self, keycode):
        key = str(keycode).lower()

        if keycode == "]":
            self.logger.warning(
                colored(
                    "[MUJOCO SIM] Keyboard ']' starts policy.",
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
        key_raw = str(cur_key)
        key = key_raw.lower()

        if key == "a":
            self.logger.warning(
                colored(
                    "[MUJOCO SIM] Joystick A starts policy.",
                    "yellow",
                )
            )
            super().handle_joystick_button(cur_key)
            super()._print_control_status()
            return

        if key == "start":
            self._handle_start_motion_clip()
        else:
            super().handle_joystick_button(cur_key)
        super()._print_control_status()

    def _capture_robot_yaw_offset(self):
        robot_state_data = self.interface.get_low_state()
        if robot_state_data is None:
            self.robot_yaw_offset = 0.0
            self.logger.warning("Unable to capture robot yaw offset - missing robot state.")
            return

        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)
        yaw = self._quat_yaw(robot_ref_ori)
        self.robot_yaw_offset = yaw
        self.logger.info(colored(f"Robot yaw offset captured at {np.degrees(yaw):.1f} deg", "blue"))

    def _capture_motion_yaw_offset(self, ref_quat_xyzw_0: np.ndarray) -> float:
        self.motion_yaw_offset = self._quat_yaw(xyzw_to_wxyz(ref_quat_xyzw_0))
        self.logger.info(colored(f"Motion yaw offset captured at {np.degrees(self.motion_yaw_offset):.1f} deg", "blue"))

    def _remove_yaw_offset(self, quat_wxyz: np.ndarray, yaw_offset: float) -> np.ndarray:
        if abs(yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw_offset)).reshape(1, 4)
        yaw_quat = np.broadcast_to(yaw_quat, quat_wxyz.shape)
        return quat_mul(yaw_quat, quat_wxyz)

    def __del__(self):
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
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    @staticmethod
    def _quat_yaw(quat_wxyz: np.ndarray) -> float:
        quat_flat = quat_wxyz.reshape(-1, 4)[0]
        _, _, yaw = quat_to_rpy(quat_flat)
        return float(yaw)
