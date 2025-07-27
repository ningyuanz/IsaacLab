# H1踢足球训练

## 概述
基于 `DirectRLEnv` 构造了一个训练 Humanoid（H1）踢足球的小项目。把踢足球简化为类射门任务，没有实现更复杂的课程。机器人需跑到足球位置，并将球踢向随机采样的目标方向。实现参考了 Isaac Lab 官方的人形机器人和四足机器人locomotion示例。

目前环境能够跑通。时间关系只简单调了下现有的参数。

实现时候的一些想法和思路放在了注释和TODO里面。

## Usage
将 `soccer` 文件夹放到 `source/isaaclab_tasks/isaaclab_tasks/direct/`.
### 训练
`./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-H1-Soccer-Direct-v0 --headless`
### 推理
`./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-H1-Soccer-Direct-v0 --num_envs 32 --checkpoint {PATH_TO_CHECKPOINT}`

## 一些实现细节
- **环境设置**：`H1SoccerEnv` 继承自 `DirectRLEnv`，场景中集成了H1（`H1_MINIMAL_CFG`）和足球（`RigidObjectCfg`）。使用平面作为地形。通过接触传感器检测机器人链接与足球的碰撞（但只实现了一个 `filtered contact sensor` 的 wrapper 类，还没来得及将检测结果用于训练）。
- **任务定义**：任务简化为射门问题。每个环境在 XY 平面上均匀采样 `[0, 2π]` 的随机目标射门方向。足球出生点固定未作随机化，可通过`H1SoccerEnvCfg.soccer_init_pos`进行配置。机器人需跑到球旁，并将其踢向目标射门方向。
- **观测**：本体感觉观测直接用了官方例子里面的躯干位置、速度、角速度、关节状态以及相对于目标的朝向等。足球相关观测包括足球在躯干和双脚局部体坐标系中的位置和速度，模拟人类踢球的感知（privileged state, 未考虑实际中机器人怎么感知到球）。
- **奖励**：
  - **运动奖励**：改编自运动示例，鼓励直立姿态、朝向对齐和能量效率。这里将原有的极远处的 target 替换成了足球的出生位置。原有官方例子用了基于欧式距离的potenrial-based的reward shaping。但踢球还需考虑朝向，至少一开始要先学会正面射门，所以是SE（2）空间中两点的距离，感觉后续可以用类似于Dubin距离优化potential奖励。
  - **射门奖励**：将 `direct.humanoid` 任务通过归一化向量的点积，奖励踢球后足球运动位移向量与目标方向的对齐。
- **重置与终止**：原有官方示例基础上，添加了足球显著移动（>1m），episode ends。重置时轻度随机化机器人root state，并将足球置于固定偏移位置。

任务复杂，跑通为主，精细的课程、reward shaping、AMP、特权训练等等都还没有。

## 结果
完全没学会，虽然毫不意外。一开始球的初始距离过近，学会了直接倒下用头撞球。后面把球移动到更远的地方，鼓励边学走路边学踢球，看奖励有上涨，但是是极致的reward hacking，跳到球附近倒下用头和身体撞球。效果见视频。

