# Pick → shoot → recovery → loco (sim2sim)

The new entrypoint is `run_three_policy.py`. It reads three WBT models in order
`pick,shoot,recovery`; loco is loaded separately from the existing local model.
Models are bundled under `src/holosoma_inference/holosoma_inference/onnx/` and
reference NPZ files under the adjacent `motions/` directory. Metadata reference
paths resolve to bundled files with the same basename first, falling back to the
original path only when no bundled file exists. Missing references raise an error.
Reference resolution is independent of the working directory; ONNX metadata is unchanged.

## Simulator terminal

```bash
cd /home/sustech/holosoma
source /opt/ros/humble/setup.bash
source /home/sustech/.holosoma_deps/miniconda3/bin/activate hsmujoco_pick
export PYTHONPATH=/home/sustech/holosoma/src/holosoma_inference:/home/sustech/holosoma/src/holosoma:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sustech/.holosoma_deps/miniconda3/envs/hsmujoco_pick/lib
python src/holosoma_inference/holosoma_inference/run_picking_sim.py robot:g1-29dof \
  --simulator.config.bridge.interface lo
```

The default picking model is the package-local `onnx/pick_17000.onnx`.
The simulator reads its embedded training configuration to obtain the motion,
ball placement, and support plate dimensions. Override with
`--picking.model-path /path/to/pick.onnx` if changing the pick model.

## Policy terminal

```bash
cd /home/sustech/holosoma
source /opt/ros/humble/setup.bash
source /home/sustech/.holosoma_deps/miniconda3/bin/activate hsinference
export PYTHONPATH=/home/sustech/holosoma/src/holosoma_inference:/home/sustech/holosoma/src/holosoma:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sustech/.holosoma_deps/miniconda3/envs/hsinference/lib/python3.10/site-packages/lib
python3 src/holosoma_inference/holosoma_inference/run_three_policy.py inference:g1-29dof-wbt-160 \
  --task.model-path "src/holosoma_inference/holosoma_inference/onnx/pick_17000.onnx,src/holosoma_inference/holosoma_inference/onnx/shoot_82000.onnx,src/holosoma_inference/holosoma_inference/onnx/recover_39999.onnx" \
  --task.no-use-joystick --task.use-sim-time --task.rl-rate 50 --task.interface lo
```

The shooting path above uses the current `shoot_82000.onnx` model. Model metadata, not hardcoded joint
angles, determines each WBT strategy's action offset and scale.

## Controls, in order

1. Policy terminal: **Enter** — smooth two-second ramp to pick reference frame 0.
2. MuJoCo window: **8** as needed — lower the robot onto its feet; wait for it to settle.
3. MuJoCo window: **B** — place/reset the basketball and small platform near the feet.
4. Policy terminal: **]** — enable pick inference with motion frame 0 pinned.
5. MuJoCo window: **9** once — disable the robot gantry; verify it is disabled.
6. Policy terminal: **s** — start pick tracking.
7. Policy terminal: **r** — toggle recording; **o** stops to the measured-pose hold.

The ball tether stays disabled for all stages. The robot gantry remains a
separate mechanism. Switching strategies never teleports or resets the ball.

## Automatic transitions

- Pick → shoot: begin comparing measured joint positions to shooting frame 0
  after pick timestep **20**; require mean absolute error ≤ **0.22 rad** and
  maximum absolute error ≤ **0.55 rad**. One qualifying sample triggers the
  switch, using the same default conditions as shoot → recovery.
- Shoot → recovery: keep the existing recovery pose-matching thresholds and
  minimum shooting timestep.
- Recovery → loco: at recovery timestep **80**.

Match settings can be changed via `--pick-match-start-frame`,
`--pick-mean-abs-threshold`, `--pick-max-abs-threshold`, and
`--pick-match-required-frames`. Reference indexing is clamped to the last
available frame independently of motion time. If pick finishes without a match,
it continues tracking the last frame for up to five seconds and then stops to a
hold at the measured joint pose, logging a warning without exiting or forcing a
switch. Shooting uses the same measured-pose hold if the recovery pose is not
reached within five seconds after its reference ends. This is a simulation
position-hold fallback, not a guarantee of physical balance or safe real deployment.

## Parameter and recording contracts

Each WBT strategy loads its own metadata KP/KD, training default joint angles,
action scale, and URDF. The reference frame-0 pose is the startup/match pose;
training default joint angles are the action offset and `dof_pos` zero point.
Pick uses real simulated ball position and zero goal. Shoot uses ball position
and the configured goal. Recovery uses zero ball/goal observations.

Loco retains the existing dual-runner default pose, metadata gains, and legacy
per-joint scale. Its ONNX does not embed a training action scale, so that scale
is not claimed to be independently verified against the original training run.

Recordings are `logs/record/three_policy_*.txt`, JSON Lines. The header lists all
four models. Frames contain role, motion time, matching errors, actual command
gains including gain multipliers, action offset/scale, raw/scaled actions,
observations, ball state and **PD torque estimate**:

`KP * (q_target - q) + KD * (dq_target - dq) + tau_ff`

This estimate is not a measured motor torque and is not torque-clipped.

## Verification

`src/holosoma_inference/tests/test_three_policy.py` runs real ONNX inference
with a simulated communication interface, checking parameter handoffs, strict
matching boundaries, all automatic state transitions and torque records.
`tests/smoke_picking_sim.py` is a bounded headless MuJoCo scene smoke test.
Controlled transition tests do not prove the robot physically picked up and
threw the ball. The new entrypoint currently rejects real deployment and
physical network interfaces; the existing sim2real/FK entrypoints are unchanged.
