import time

import tyro

from holosoma.utils.sim_utils import DirectSimulation, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG
from holosoma_inference.picking_scene import PickingReference
from holosoma_inference.run_picking_sim import DEFAULT_PICK, MuJoCo, PickingRunConfig


def main():
    config = tyro.cli(PickingRunConfig, config=TYRO_CONIFG, args=["robot:g1-29dof", "--training.headless", "True",
                      "--simulator.config.bridge.interface", "lo"])
    from dataclasses import replace
    import mujoco
    import numpy as np
    from holosoma_inference.picking_scene import yaw

    reference = PickingReference(DEFAULT_PICK)
    MuJoCo.reference = reference
    delta = -yaw(reference.root_pose[3:])
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, np.array([np.cos(delta / 2), 0, 0, np.sin(delta / 2)]), reference.root_pose[3:])
    config = replace(config, simulator=replace(config.simulator, _target_="holosoma_inference.run_picking_sim.MuJoCo"),
        robot=replace(config.robot, init_state=replace(config.robot.init_state,
            pos=[0, 0, float(reference.root_pose[2])], rot=quat[[1, 2, 3, 0]].tolist(),
            default_joint_angles=dict(zip(reference.joint_names, reference.joint_pos.tolist())))))
    env, device, app = setup_simulation_environment(config, device=config.device)
    with DirectSimulation(config, env, device, app) as simulation:
        sim = simulation.simulator
        np.testing.assert_allclose(sim.root_data.qpos[sim.dof_qpos_addrs],
            [config.robot.init_state.default_joint_angles[name] for name in sim.dof_names], atol=1e-7)
        started = time.monotonic()
        placed = False
        released = False
        lowered = False
        while time.monotonic() - started < 14:
            elapsed = time.monotonic() - started
            if elapsed > 5 and not lowered:
                for _ in range(5):
                    sim._key_callback(56)
                lowered = True
            if elapsed > 5.8 and not placed:
                sim._key_callback(66)
                placed = True
            if elapsed > 7.5 and not released:
                sim._key_callback(57)
                released = True
            sim.simulate_at_each_physics_step()
            time.sleep(1 / config.simulator.config.sim.fps)
        assert sim.picking_scene.placed
        assert not sim.object_tether_enabled
        assert sim.virtual_gantry is None or not sim.virtual_gantry.enabled
        print("HEADLESS_SCENE_PASS", sim.root_data.qpos[sim.picking_scene.qpos].tolist(), flush=True)


if __name__ == "__main__":
    main()
