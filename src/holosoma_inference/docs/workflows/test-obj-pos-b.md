# 只读测试 pelvis → torso → obj_pos_b

脚本：`src/holosoma_inference/holosoma_inference/test_obj_pos_b.py`。
不加载 ONNX session、不设置 PR、不发送 lowcmd。SDK2Py 只创建 `rt/lowstate` 的 HG
`ChannelSubscriber`，不创建 C++ 双向 binding、命令 publisher 或 RPC 控制客户端；
只读 q/dq/IMU/tick。
动捕读取 pelvis 与篮球中心，使用与部署文件共享的 `torso_from_pelvis()` 和相同 Pinocchio FK。
脚本不会让机器人站立、进入阻尼或改变当前控制状态。

## 启动

先启动机器人状态通信和真实动捕 ROS2 发布端，保证局域网/ROS_DOMAIN_ID 可互通。

```bash
cd /home/sustech/holosoma
source /opt/ros/humble/setup.bash
source /home/sustech/.holosoma_deps/miniconda3/bin/activate hsinference
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sustech/.holosoma_deps/miniconda3/envs/hsinference/lib/python3.10/site-packages/lib
python3 src/holosoma_inference/holosoma_inference/test_obj_pos_b.py \
  --interface eth0 \
  --pelvis-topic /pelvis/pose \
  --ball-topic /basketball/pose \
  --mocap-unit mm
```

将 `eth0` 改为实际连接 G1 的网卡。若 topic 为 `/pelvis/pos`，修改对应参数。
单位可选 `mm/cm/m`，必须与发布端实际单位一致。
`--urdf` 默认本地 `g1_29dof.urdf`；应与部署所用机器人几何一致，若有修改请显式提供相同版本。
pelvis 动捕坐标系需要已经与 URDF pelvis link 原点/xyz 轴一致，不再添加旧的经验偏移。

## 按键

| 键 | 行为 |
|---|---|
| r | 开始/停止录制；刚开始默认阶段1 |
| 1 | 脚边静止 |
| 2 | 沿 pelvis Z 上下移动 |
| 3 | 沿 pelvis X 前后移动 |
| 4 | 沿 pelvis Y 左右移动 |
| q、Esc、Ctrl-C | 退出并关闭记录 |

推荐：`r → 1 → 静止数秒 → 2 → 上下移动 → 3 → 前后移动 → 4 → 左右移动 → r`。
每次阶段按键创建独立 segment，即使重复选择同一阶段也可分别统计。
无交互终端时可用 `--record-on-start --max-seconds 20` 有界自动记录。

## 两个相对位置

- `ball_pos_pelvis = R_pelvis.T @ (ball_world - pelvis_world)`：只依赖动捕。
- `obj_pos_b = R_torso.T @ (ball_world - torso_world)`：pelvis 动捕＋SDK 实测关节角＋URDF FK。

当沿 pelvis X/Y/Z 移动时，优先检查 `ball_pos_pelvis` 的相应轴是否为主要变化方向。
腰部 yaw/roll/pitch 不为零时，pelvis 轴不等于 torso 轴，`obj_pos_b` 多轴变化可能正确。
程序使用 SDK 实测关节角，不使用假设的腰部零位。记录 `waist_yaw_roll_pitch`（rad）。

## 记录格式

输出：`logs/record/obj_pos_b_test_日期_时间.txt`，严格 JSON Lines。

- `metadata`：参数、单位、URDF SHA-256、joint names/mapping、只读标志、时间基准。
- `pose`：每条实际进入本脚本回调的 ROS 消息，包括无效/重复消息；stream、sequence、
  原始位置和 quaternion、米制位置、header stamp_ns/frame_id、单调时钟接收时间、accepted/reason。
- `sdk`：每条进入 SDK 订阅回调的 lowstate，包含序号、tick、state、回调接收时刻、
  tick首次被观察更新的时刻、valid/reasons；`sdk_poll` 是独立的主循环采样记录。
- `calculation`：pelvis/ball/SDK 消息序号、配对时间差、FK torso 位姿、两个相对位置、有效性。
  无效计算使用 `valid=false` 和原因，相对位置为 null，不用零向量冒充。
- `stage_event`：阶段标记和时间。
- `rates`：每秒频率快照，`scope=process_lifetime` 的 streams 为进程累计统计；
  `segment_streams` 为当前阶段统计，各流附带 first/last_received_at。
- `summary`：每个 segment 的有效样本数、位置均值/标准差/范围、频率与时间戳异常计数。

录制开始和阶段切换后清空配对缓存，确保 calculation 引用的是本次记录/当前阶段中保存的消息。
跨越切换边界才取得处理锁的旧回调仍保存为原始行，但标记 `admitted_to_phase=false`，
不参与新阶段的配对和统计。
异常浮点原始值会保留为字符串，例如 `"nan"`，使文件仍能按标准 JSON 解析。

## 时间配对和频率含义

ROS 使用独立接收线程和 BEST_EFFORT/depth=100 队列，兼容常见 reliable/best-effort 发布端。
保存所有本脚本实际收到的回调；不声称捕获了网络中全部发送包。队列溢出或网络丢包仍可能发生。
SDK 默认100Hz采样；ROS接收不受这一循环频率限制。

计算时选最新篮球样本，并从 pelvis 缓存选源时间戳最接近的一条；默认最大源时间差0.04s。
关节状态按 SDK 回调观察到 tick 更新的本机时刻选最近样本，默认最大差0.05s，
先选最近，再检查有效性，不跳过最近的无效消息而偷偷使用别的有效消息。
主循环反复读取同一个tick不会刷新测量更新时间，也不会绕过0.05s配对检查。
没有将 Unitree tick 映射到动捕源时钟，因此不是严格的硬件同步。接收时间差会记录以供评估。

默认超过0.25s无新数据、源时间戳过旧/未来、global frame不同或SDK tick停更均标记无效。
两电脑时钟不同步会使源年龄检查失败，原始数据仍保存；不要把钟差直接当网络延迟。

频率分开统计：

| 名称 | 实际含义 |
|---|---|
| pelvis/ball `receive_hz` | 回调接收/处理平均频率；超过1秒没有回调时显示0 |
| pelvis/ball `source_hz` | 根据正的相邻 header 时间间隔推算的源更新频率 |
| sdk_poll | 本脚本轮询SDK的频率 |
| sdk_receive | SDK订阅回调的实际接收/处理频率，不保证等于机器人发送频率 |
| sdk_observed_updates | 采样中观察到 tick 改变的频率，不能当成DDS原始发布频率 |
| valid_computations | 新的动捕配对通过校验后实际计算的频率 |

记录间隔均值/最大值、重复/倒退时间戳、超过100ms的接收/源间隔。
没有可靠源序号，因此大间隔不能精确换算丢帧数。实时报告为累计平均，segment summary
保留该段的历史平均，不会因切到下一段后等待而把前段频率误报成零。

## 软件验证

`tests/test_obj_pos_b_probe.py` 用假 Unitree SDK 检查控制调用为零，
验证零位/非零腰部FK、世界平移旋转不变性、单位、过期/错帧/时间差、频率和录制阶段。
包含真实本机 ROS2 合成发布/订阅测试，但不连接真实G1；不等于现场动捕标定正确性的证明。
代码修改和应用记录文件限定在本仓库；ROS/Python 等第三方库的运行缓存不是文件系统沙箱。
