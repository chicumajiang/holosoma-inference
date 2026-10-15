# 三策略真机部署：deploy_tri_policy.py

入口：`src/holosoma_inference/holosoma_inference/deploy_tri_policy.py`。
共享 `run_three_policy.py` 的 pick → shoot → recovery → loco 状态切换和 WBT 模型参数，
不使用 MuJoCo 的 `/object_state` 或 `/torso_link_pos` 作为真机观测来源。

## 三种模式

| 模式 | ONNX 推理 | 设置 PR | 发送 lowcmd | Enter 行为 |
|---|---|---|---|---|
| read-only（默认） | 不加载 ONNX session | 否 | 否 | 不移动机器人 |
| dry-run | 加载三个 WBT 和 loco，执行实际推理及限制 | 否 | 否 | 只计算初始姿态过渡 |
| armed | 同 dry-run | 经 Enter 和状态检查后允许 | 所有检查通过后允许 | 真实缓升到 pick default pose |

read-only/dry-run 不会通过本入口发送电机命令；记录文件可以写入。
这不约束其他程序、机器人原生控制器或用户直接调用原生 SDK。

## 输入契约

- 连接 G1 的真实网卡，例如 `eth0`；拒绝 `lo`、sim time、非 50 Hz 和非零 Unitree DDS domain。
- `/pelvis/pose`：`geometry_msgs/msg/PoseStamped`，刚体原点及 xyz 轴已经由你在动捕中
  定义为 URDF **pelvis link** 的原点及轴。代码不会再增加旧的 1 cm / 3.4 cm 经验偏移。
- `/basketball/pose`：同类型，position 必须是篮球中心。篮球 orientation 不用于 actor，
  因此允许位置有效但没有 orientation 的数据。
- 默认输入单位 **mm**；通过 `--deploy-mocap-unit mm/cm/m` 明确选择，转换后统一为米。
- 两个消息必须有相同非空 global `frame_id`、有效且同步的源时间戳。
  默认接收年龄/源时间戳年龄 ≤0.25 s，球与 pelvis stamp 差 ≤0.04 s。
  源时间戳必须推进；重放旧时间戳不会刷新缓存。
- 使用 VRPN 客户端本机时间戳或同步好服务器时钟；时钟不一致会拒绝启动。
- Unitree lowstate 来自实际 SDK binding 的 `tick`、IMU、q/dq。启用前必须观察到 tick
  推进；停止更新超过 0.25 s 会停止控制。读取耗时检查与 tick freshness 是不同检查。

位置关系：pelvis 完整动捕位姿作为 Pinocchio free-flyer 根，结合 DDS 腰部关节角计算
torso，再计算 `obj_pos_b = R_torso^T * (p_ball - p_torso)`。
这个计算不要求动捕世界原点与训练原点相同。局部标定、轴方向和单位仍需要你现场验证。

## 篮球放置门禁

目标从 **pick 模型 metadata 指向的 motion 第一帧**计算出篮球相对 torso 的位置。
dry-run/armed 检查当前实际 `obj_pos_b` 与目标的每轴误差，默认每轴 ±3 cm，
并要求机器人达到 pick metadata 的 default joint pose（最大误差 ≤0.10 rad，
最大关节速度 ≤0.3 rad/s），连续有效至少 0.5 s。

每秒报告，例如：

```text
篮球放置：不行，沿 torso 坐标轴移动 x-10.0cm y+6.0cm z+0.0cm。
篮球放置：行，可以按 ] 开启 pick policy。
```

正负号是 **目标 − 实测**：x+ 向 torso +X，y+ 向 +Y，z+ 向 +Z。
不是动捕 global 轴。`--deploy-tolerance 0.02` 可改为每轴 ±2 cm。
误差合格但姿态不稳定、数据缺失/过期/不同步均不能通过。
`s` 开始 tracking 时再次检查。`o` 不能绕过 Enter 和初始过渡完成的要求。

当前 pick 的 reference 初始篮球中心高度约 **0.216 m**，训练/仿真有约 **0.097 m**
的小平台。直接放在地上可能在 z 方向差约 10 cm，门禁会如实拒绝；使用相应平台或重新
训练一致的地面捡球参考，不能靠屏蔽 z 误差绕过。

read-only 的 `--deploy-reference` 默认指向包内 `motions/basketball_013_pick_extend_ball_fixed_new.npz`，其位置检查是参考预检；
不加载 ONNX，也不证明它与以后选择的其他 pick 模型合同一致。最终准入以 dry-run/armed
实际 pick metadata 为准。

模型 metadata 中的 reference 路径统一优先解析为包内 `motions/` 下的同名文件，
不存在时才使用 metadata 原路径；两处都不存在则报错。无需修改 ONNX metadata。
`motions/` 相对于 Python 包的位置解析，与启动目录无关。显式指定的
`--deploy-reference` 按指定路径读取，不会被同名包内文件覆盖。

## 环境

```bash
cd /home/sustech/holosoma
source /opt/ros/humble/setup.bash
source /home/sustech/.holosoma_deps/miniconda3/bin/activate hsinference
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sustech/.holosoma_deps/miniconda3/envs/hsinference/lib/python3.10/site-packages/lib
```

先启动真实动捕的 ROS2 发布程序。下面的 `eth0` 替换成实际 G1 网卡，
`/pelvis/pose` 可通过 `--deploy-pelvis-topic /pelvis/pos` 修改，消息类型不变。

### read-only

```bash
python3 src/holosoma_inference/holosoma_inference/deploy_tri_policy.py \
  inference:g1-29dof-wbt-160 \
  --deploy-mode read-only --deploy-mocap-unit mm \
  --task.interface eth0 --task.rl-rate 50 --task.no-use-joystick \
  --deploy-record --deploy-max-iterations 250
```

不需要模型路径。自动记录 `logs/record/tri_readonly_*.txt`，包含原始/米制动捕数据、
源/接收时间戳、frame_id、pelvis/torso quaternion、lowstate tick、关节序号和位置检查。
read-only 的 Enter 不会把机器人移到初始姿态，需人工/其他已验证控制方式准备。

### dry-run 与 armed

三个 WBT 按 pick,shoot,recovery 顺序传入。loco 自动使用当前本地模型。
**loco ONNX 没有 action_scale metadata**，必须显式提供 `--deploy-loco-action-scale`。
可以给一个统一数值或29个逗号分隔值；应使用该 loco 模型训练时的实际值。
当前 sim2sim 使用的 legacy 向量可从
`config/config_values/robot_init.py:G1_29DOF_PER_JOINT_ACTION_SCALE` 查询，
它没有因此被证明是原训练合同。不要把示例 scale 当作已经验证的真机参数。

```bash
MODEL_DIR="src/holosoma_inference/holosoma_inference/onnx"
MODELS="$MODEL_DIR/pick_17000.onnx,$MODEL_DIR/shoot_82000.onnx,$MODEL_DIR/recover_39999.onnx"
# 用已经确认的 loco 训练 scale 替换这一变量，可以是单值或者29维向量。
LOCO_SCALE="已确认的数值"
python3 src/holosoma_inference/holosoma_inference/deploy_tri_policy.py \
  inference:g1-29dof-wbt-160 \
  --task.model-path "$MODELS" --deploy-mode dry-run \
  --deploy-loco-action-scale "$LOCO_SCALE" \
  --deploy-mocap-unit mm --task.interface eth0 --task.rl-rate 50 \
  --task.no-use-joystick --deploy-record
```

dry-run 只计算，机器人不会跟着 Enter 过渡；需要实测姿态本来已到初始位姿，
才能通过放球门禁并用 `]`、`s` 做真实输入推理。
每秒报告最终限幅 command、有效 KP/KD、PD 力矩估计。

armed 使用同一命令，把模式改为 `--deploy-mode armed`。要求交互终端并输入
`I_UNDERSTAND_THIS_COMMANDS_A_REAL_ROBOT`，然后：

1. Enter：用 pick 模型增益在默认10秒内平滑过渡到 pick default pose。
2. 把篮球放到脚边符合训练的位置和高度，等待每秒报告“行”。
3. `]`：启用 pick policy，第0帧暂停。
4. `s`：再次校验篮球放置后开始 tracking；自动 pick → shoot → recovery → loco。
5. `r`：记录开关；`o`：取消 tracking 并向实测姿态限速保持。
   Esc 或 Ctrl-C 关闭部署；键盘在控制循环中轮询，不使用后台监听线程。

## 控制限制与退出

逐关节校验模型增益/scale、URDF requested/bounded 目标范围、每周期目标增量默认
0.02 rad、最终目标与实测误差默认0.35 rad、实测速度默认8 rad/s、预测 PD 力矩不超过
模型 effort limit。LOCO 使用独立配置的 G1 硬件 URDF（`--deploy-loco-urdf`），
其 gains 来自 loco ONNX，scale来自显式输入，而不是借 pick/recovery 的限位/effort。

控制循环与键盘异常会锁定 fault，不再普通发送。退出曾启用写入的 armed 时，会请求
KP=0、KD最多2的阻尼命令（按当前速度和硬件 effort 限制），然后锁闭 Python 写通道。
如果退出时不能验证 lowstate 新鲜有效，则请求零增益，而不是用旧速度估计阻尼。
这是请求，**不是电机反馈确认，也不能保证站立平衡**。SDK 没暴露 native stop/watchdog API，
无法由 Python 证明 native writer 停止或命令持久性；现场支撑和硬件急停仍然必要。
不对真实硬件自动运行 armed 验收。

`r` / `--deploy-record` 输出 `three_policy_*.txt`，普通 frame 后增加 tracking 行，
包含门禁结果、单位、pelvis/ball原始位姿、FK torso、yaw offsets及是否启用写入。
`torque_pd_estimate` 是控制律估计值。真机 `measured_joint_torque` 来自 Unitree lowstate
的 `motor.tau_est`（固件估计，不是独立力矩传感器），`torque_received` 标识是否可用；
不再依赖仿真的 `/joint_torque`。read-only 记录 `motor_tau_est`，不读取模型 KP/KD。
`object_velocity_w` 没有真机速度来源，tracking 行明确标识 `ball_velocity_available=false`。

## 代码验收

`tests/test_deploy_tri_policy.py` 使用真实 ONNX 和 fake Unitree binding：记录真实调用
`set_control_mode` / `write_low_command` 的次数，验证非armed无调用、Enter门禁、错位/缺失/
过期/不同步动捕、坐标平移不变性、源时间戳重放、tick冻结、退出锁闭和独立loco合同。
这是软件/模拟硬件测试，不是完整真机动作或 native writer 行为的实测证明。
