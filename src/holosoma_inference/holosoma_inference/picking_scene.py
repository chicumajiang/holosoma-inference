from __future__ import annotations

import json

import mujoco
import numpy as np
import onnx

from holosoma_inference.motion_paths import resolve_motion_path


def yaw(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class PickingReference:
    def __init__(self, model_path):
        metadata = {prop.key: prop.value for prop in onnx.load(str(model_path)).metadata_props}
        config = json.loads(metadata["experiment_config"])
        motion = config["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]
        plate = config["robot"]["object"]["support_plate"]
        if not plate["enabled"] or motion["body_name_ref"] != ["torso_link"]:
            raise ValueError("Picking requires a torso reference and enabled support plate.")
        self.plate_size = np.asarray(plate["size"], dtype=np.float64)
        self.plate_pos = np.asarray(plate["position"], dtype=np.float64)
        self.plate_color = np.asarray(plate["color"], dtype=np.float64)
        with np.load(resolve_motion_path(motion["motion_file"]), allow_pickle=False) as data:
            index = data["body_names"].tolist().index("torso_link")
            self.torso_pos = data["body_pos_w"][0, index].copy()
            self.torso_quat = data["body_quat_w"][0, index].copy()
            self.ball_pos = data["object_pos_w"][0].copy()
            self.ball_quat = data["object_quat_w"][0].copy()
            self.root_pose = data["joint_pos"][0, :7].copy()
            self.joint_names = data["joint_names"].tolist()
            self.joint_pos = data["joint_pos"][0, 7:].copy()
        if not all(np.isfinite(value).all() for value in vars(self).values() if isinstance(value, np.ndarray)):
            raise ValueError("Non-finite picking reference.")
        if any(value.shape != (3,) for value in (self.torso_pos, self.ball_pos, self.plate_pos, self.plate_size)):
            raise ValueError("Picking positions and plate size must contain three values.")
        model_names = json.loads(metadata["dof_names"])
        if (np.any(self.plate_size <= 0) or len(self.joint_names) != 29 or len(set(self.joint_names)) != 29
                or set(self.joint_names) != set(model_names) or self.joint_pos.shape != (29,)):
            raise ValueError("Invalid platform dimensions or picking joint configuration.")
        for quat in (self.torso_quat, self.ball_quat, self.root_pose[3:]):
            if quat.shape != (4,) or np.linalg.norm(quat) < 1e-8:
                raise ValueError("Invalid reference quaternion.")
            quat /= np.linalg.norm(quat)

    def targets(self, torso_pos, torso_quat):
        delta = yaw(torso_quat) - yaw(self.torso_quat)
        c, s = np.cos(delta), np.sin(delta)
        rotation = np.array([[c, -s], [s, c]])
        ball, plate = self.ball_pos.copy(), self.plate_pos.copy()
        ball[:2] = torso_pos[:2] + rotation @ (ball[:2] - self.torso_pos[:2])
        plate[:2] = torso_pos[:2] + rotation @ (plate[:2] - self.torso_pos[:2])
        quat = np.array([np.cos(delta / 2), 0, 0, np.sin(delta / 2)])
        ball_quat = np.empty(4)
        mujoco.mju_mulQuat(ball_quat, quat, self.ball_quat)
        return ball, ball_quat / np.linalg.norm(ball_quat), plate, quat


class PickingScene:
    def __init__(self, model, data, reference, torso_name, ball_name, joint_name):
        self.model, self.data, self.reference = model, data, reference
        self.ball_id = model.body(ball_name).id
        self.torso_id = model.body(torso_name).id
        self.plate_id = model.body("support_plate").id
        joint = model.joint(joint_name).id
        self.qpos = slice(model.jnt_qposadr[joint], model.jnt_qposadr[joint] + 7)
        self.qvel = slice(model.jnt_dofadr[joint], model.jnt_dofadr[joint] + 6)
        self.mocap_id = model.body_mocapid[self.plate_id]
        self.geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, [self.ball_id, self.plate_id]))
        self.body_ids = np.array([self.ball_id, self.plate_id])
        self.rgba = model.geom_rgba[self.geom_ids].copy()
        self.contype = model.geom_contype[self.geom_ids].copy()
        self.conaffinity = model.geom_conaffinity[self.geom_ids].copy()
        self.body_contype = model.body_contype[self.body_ids].copy()
        self.body_conaffinity = model.body_conaffinity[self.body_ids].copy()
        self.gravcomp = model.body_gravcomp[self.ball_id]
        self.placed = False

    def clear_motion(self):
        for value in (self.data.qvel, self.data.qacc, self.data.qacc_warmstart, self.data.qfrc_applied):
            value[self.qvel] = 0
        self.data.xfrc_applied[self.ball_id] = 0

    def hide(self):
        self.placed = False
        self.model.geom_rgba[self.geom_ids, 3] = 0
        self.model.geom_contype[self.geom_ids] = 0
        self.model.geom_conaffinity[self.geom_ids] = 0
        self.model.body_contype[self.body_ids] = 0
        self.model.body_conaffinity[self.body_ids] = 0
        self.model.body_gravcomp[self.ball_id] = 1
        self.data.mocap_pos[self.mocap_id] = [10, 10, -10]
        self.data.mocap_quat[self.mocap_id] = [1, 0, 0, 0]
        self.park()
        mujoco.mj_forward(self.model, self.data)

    def park(self):
        if not self.placed:
            self.data.qpos[self.qpos] = [10, 10, -10, 1, 0, 0, 0]
            self.clear_motion()

    def place(self):
        mujoco.mj_forward(self.model, self.data)
        ball, ball_quat, plate, quat = self.reference.targets(
            self.data.xpos[self.torso_id], self.data.xquat[self.torso_id])
        self.data.qpos[self.qpos] = np.concatenate([ball, ball_quat])
        self.data.mocap_pos[self.mocap_id] = plate
        self.data.mocap_quat[self.mocap_id] = quat
        self.model.geom_rgba[self.geom_ids] = self.rgba
        self.model.geom_contype[self.geom_ids] = self.contype
        self.model.geom_conaffinity[self.geom_ids] = self.conaffinity
        self.model.body_contype[self.body_ids] = self.body_contype
        self.model.body_conaffinity[self.body_ids] = self.body_conaffinity
        self.model.body_gravcomp[self.ball_id] = self.gravcomp
        self.clear_motion()
        mujoco.mj_forward(self.model, self.data)
        self.data.qacc_warmstart[self.qvel] = 0
        self.placed = True
        return {"ball_pos": ball.tolist(), "plate_pos": plate.tolist()}
