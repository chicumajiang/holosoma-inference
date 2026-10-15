from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(PACKAGE_ROOT.parent), str(PACKAGE_ROOT.parents[1] / "holosoma")]

import mujoco
import numpy as np
import tyro
from loguru import logger

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.simulator.mujoco.mujoco import (
    MuJoCo as BaselineMuJoCo,
    _mujoco_attach_object_mjcf_to_world_spec,
    _mujoco_make_object_mjcf_xml,
    _mujoco_read_largebox_urdf_params,
)
from holosoma.utils.sim_utils import DirectSimulation, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG
from holosoma_inference.picking_scene import PickingReference, PickingScene, yaw


DEFAULT_PICK = str(PACKAGE_ROOT / "onnx/pick_17000.onnx")


class MuJoCo(BaselineMuJoCo):
    reference: PickingReference

    def _setup_scene(self):
        terrain = self.terrain_manager.get_state("locomotion_terrain")
        if terrain.mesh_type not in ["none", "fake"]:
            self.scene_manager.add_terrain(terrain, self.training_config.num_envs)
            self.scene_manager.add_lighting()
            self.scene_manager.add_materials()
        self.scene_manager.add_robot(terrain, self.robot_config, xml_filter=self.simulator_config.robot_mjcf_filter)
        root = Path(__file__).resolve().parents[2] / "holosoma/holosoma/data"
        params = _mujoco_read_largebox_urdf_params(root / "motions/g1_29dof/whole_body_tracking/objects_largebox.urdf")
        xml = _mujoco_make_object_mjcf_xml(self.object_name, (10, 10, -10), params)
        if not _mujoco_attach_object_mjcf_to_world_spec(self, xml):
            raise RuntimeError("Could not attach basketball mesh.")
        body = self.scene_manager.world_spec.worldbody.add_body(name="support_plate", pos=[10, 10, -10])
        body.mocap = True
        body.add_geom(name="support_plate_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=self.reference.plate_size / 2,
                      rgba=np.concatenate([self.reference.plate_color, [1.0]]), contype=2, conaffinity=5)

    def _set_object_anchor_addressing(self):
        self.object_anchor_body_id = None
        self.object_anchor_mocap_id = None

    def prepare_sim(self):
        super().prepare_sim()
        self.root_data.qpos[self.dof_qpos_addrs] = [
            self.robot_config.init_state.default_joint_angles[name] for name in self.dof_names]
        self.root_data.qvel[self.dof_qvel_addrs] = 0
        mujoco.mj_forward(self.root_model, self.root_data)
        self.picking_request = Event()
        self.object_tether_enabled = False
        self.debug_joint_qpos_enabled = False
        self.picking_scene = PickingScene(
            self.root_model, self.root_data, self.reference, self.clean_to_prefixed_names["torso_link"],
            mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_BODY, self.object_body_id),
            mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_JOINT, self.object_joint_id))
        self.picking_scene.hide()

    def _apply_object_tether_force(self):
        self.object_tether_enabled = False
        self.root_data.xfrc_applied[self.object_body_id] = 0

    def on_episode_start(self, env_id=0):
        super().on_episode_start(env_id)
        gantry = self.virtual_gantry
        if gantry is not None and gantry.enabled and self.simulator_config.virtual_gantry.length == 0:
            ids = [i for i in range(self.root_model.nbody)
                   if (mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("robot_")]
            mass = float(self.root_model.body_mass[ids].sum())
            height = self.root_data.qpos[self.robot_qpos_addr + 2]
            gantry.length = gantry.height - height - 0.9 * mass * abs(self.root_model.opt.gravity[2]) / gantry.stiffness
            logger.info("Picking robot gantry calibrated: mass={} rest_length={}", mass, gantry.length)

    def _key_callback(self, keycode):
        if keycode == 66:
            self.picking_request.set()
        elif keycode in (85, 74, 72, 75, 84, 78, 76, 80):
            logger.info("Ball tether disabled; use B to place/reset the ball and platform.")
        else:
            super()._key_callback(keycode)

    def simulate_at_each_physics_step(self):
        if self.picking_request.is_set():
            self.picking_request.clear()
            logger.info("Picking placement: {}", self.picking_scene.place())
        self.picking_scene.park()
        if self.virtual_gantry is not None and not self.virtual_gantry.enabled:
            self.root_data.xfrc_applied[self.virtual_gantry.body_link_id] = 0
        super().simulate_at_each_physics_step()
        self.picking_scene.park()


@dataclass
class PickingOptions:
    model_path: str = DEFAULT_PICK


@dataclass(frozen=True)
class PickingRunConfig(RunSimConfig):
    picking: PickingOptions = field(default_factory=PickingOptions)


def main():
    config = tyro.cli(PickingRunConfig, config=TYRO_CONIFG)
    if config.simulator.config.name != "mujoco" or config.robot.asset.robot_type != "g1_29dof":
        raise ValueError("Picking simulator requires classic MuJoCo and G1-29DoF.")
    reference = PickingReference(config.picking.model_path)
    if set(reference.joint_names) != set(config.robot.dof_names):
        raise ValueError("Picking reference joint names do not match the selected robot.")
    from holosoma_inference.run_picking_sim import MuJoCo as SimulatorClass
    SimulatorClass.reference = reference
    root = reference.root_pose[3:]
    delta = -yaw(root)
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, np.array([np.cos(delta / 2), 0, 0, np.sin(delta / 2)]), root)
    config = replace(config, simulator=replace(config.simulator,
                     _target_="holosoma_inference.run_picking_sim.MuJoCo"),
                     robot=replace(config.robot, init_state=replace(config.robot.init_state,
                         pos=[0, 0, float(reference.root_pose[2])], rot=quat[[1, 2, 3, 0]].tolist(),
                         default_joint_angles=dict(zip(reference.joint_names, reference.joint_pos.tolist(), strict=True)))))
    env, device, app = setup_simulation_environment(config, device=config.device)
    with DirectSimulation(config, env, device, app) as simulation:
        simulation.run()


if __name__ == "__main__":
    main()
