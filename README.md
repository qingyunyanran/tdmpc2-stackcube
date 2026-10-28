# TD-MPC2 on ManiSkill3 · StackCube-v1

把官方 [TD-MPC2](https://github.com/nicklashansen/tdmpc2)（ManiSkill2 版）移植到 **ManiSkill3**，
在 `StackCube-v1` 上跑通并完成训练，结果与命令全部可复现。

**本仓库取代了旧版内容**：之前那套 CPU-only + 专家 demo warm-start 的尝试（`train_stackcube.py` /
`warmstart.py` / `warmstart2.py` / `launch_*.bat` / `tdmpc2/logs/`）成功率只有 5–10%，
已经整体删除，代码在 git 历史里仍可查。

## 结果

单 seed、`model_size=5`、20 集确定性终评（`scripts/eval_20ep.sh`）：

| 训练 | 步数 | 成功率 | 平均 return | 平均成功保持步数 |
|---|---|---|---|---|
| 从零 | 2M | **70%** (14/20) | 1313.8 | — |
| 2M 权重 warm start 续训 | +2M（合计 4M） | **80%** (16/20) | 1228.4 | 67.2 |

训练过程中的每 50k 一次 eval（10 集）完整记录在
[`results/ms3_stack_2m/eval.csv`](results/ms3_stack_2m/eval.csv) 与
[`results/ms3_stack_4m_ws/eval.csv`](results/ms3_stack_4m_ws/eval.csv)。

同一套代码在 ManiSkill2 上的对照（论文口径，4M steps）：

| 任务 | 成功率 | 平均 return |
|---|---|---|
| PickCube-v0 | 100% (20/20) | 78.0 |
| StackCube-v0 | 90% (18/20) | 2637.5 |

> 注意：MS2 与 MS3 的 dense reward 尺度不同（MS3 每步 0.27–0.43、成功奖励 8；MS2 约 26 / 15），
> 所以 return 数值不能跨后端直接比较。

## 和官方仓库相比改了什么

官方代码假设 ManiSkill2 + 旧版 gym/torch。移植到 ManiSkill3 的最小改动是 8 个文件：

| 文件 | 改动 |
|---|---|
| `tdmpc2/envs/maniskill3.py` | **新增**。ManiSkill3 后端，obs 拼接 + 动作重缩放 + success 透传 |
| `tdmpc2/envs/__init__.py` | 用 `MS_BACKEND=ms3` 在 MS2 / MS3 两个后端间切换 |
| `tdmpc2/envs/wrappers/tensor.py` | 兼容 gymnasium 1.4 的 5 元组 `step()` 转发 |
| `tdmpc2/envs/maniskill.py` | 同步修正 step 元组处理 |
| `tdmpc2/common/logger.py` | eval 指标增加 `episode_success` 列；视频落盘 |
| `tdmpc2/config.yaml` | 新增 `resume: ''`（warm start 入口） |
| `tdmpc2/train.py` | 先 `make_env` 再 `TDMPC2`，并按需 `agent.load(...)` |
| `tdmpc2/trainer/online_trainer.py` | warm start 时跳过 seed pretraining |
| `tdmpc2/tdmpc2.py` | `compile_update` 开关 |

算法本身（世界模型结构、MPPI 规划、损失、超参）**没有改动**。

## 环境

- Python 3.11 / `mani_skill 3.0.1` / `sapien 3.0.3` / `gymnasium 1.4.0`
- `torch 2.7.1+cu126` / `tensordict 0.8.3` / `torchrl 0.8.1`

**torch 版本是硬约束**：torch 2.14 的 dynamo 在 torchrl 的 TensorDict 路径上会抛
`InternalTorchDynamoError: NameError: name 'SYNTHETIC_LOCAL' is not defined`
（`torch/_dynamo/guards.py`），导致 `compile=true` 必崩；关掉 compile 只有 13–17 agent steps/s。
降到 2.7.1 后可稳定 38–39 steps/s（瓶颈是 `env.step`，约 7 ms/步）。

```bash
bash scripts/install_env.sh
```

## 复现

```bash
# 0) 环境变量（所有脚本都要求）
export TDMPC2_ROOT=/root/tdmpc2          # 本仓库根目录
export VK_ICD_FILENAMES=/root/egl_icd.json   # headless 渲染所需的 Vulkan ICD

# 1) 从零训练 2M steps
bash scripts/train_2m.sh

# 2) 20 集终评
CKPT=$TDMPC2_ROOT/logs/stack-cube/1/ms3_stack_2m/models/final.pt bash scripts/eval_20ep.sh

# 3) 用 2M 权重 warm start 再训 2M（合计 4M）
bash scripts/train_warmstart_4m.sh

# 4) 再评一次
CKPT=$TDMPC2_ROOT/logs/stack-cube/1/ms3_stack_4m_ws/models/final.pt bash scripts/eval_20ep.sh

# 5) 失败集诊断（初始位姿 → 失败原因）
CKPT=$TDMPC2_ROOT/logs/stack-cube/1/ms3_stack_4m_ws/models/final.pt python scripts/fail_diag.py
```

## ManiSkill3 适配要点

- **观测**：`(48,)` = qpos(9) + qvel(9) + tcp_pose(7) + cubeA_pose(7) + cubeB_pose(7)
  + tcp_to_cubeA(3) + tcp_to_cubeB(3) + cubeA_to_cubeB(3)。
  比 MS2 的 `(55,)` 少了 cube 的速度分量。
- **动作**：`(4,)` 归一化后重缩放到控制器的 action space；`control_freq=20`，
  `action_repeat=2` → 每集 200 env steps = 100 agent steps。
- **`episodic=false`**：抑制 `info['terminated']`，让 trainer 不因任务成功而提前截断 episode
  （与 MS2 上的复现口径一致）。
- **success 判定逐字相同**：`is_cubeA_on_cubeB & is_cubeA_static & ~is_cubeA_grasped`，
  两个后端的 reset 采样（`xy=uniform(-0.1,0.1)` + `region=[[-0.1,-0.2],[0.1,0.2]]`）也逐行同构，
  所以 MS3 的 80% 不是"任务变难"造成的。
- **warm start 的两个坑**：①`make_env(cfg)` 必须先于 `TDMPC2(cfg)`，否则
  `AttributeError: 'str' object has no attribute 'keys'`（cfg 里还没有 `obs_shape`）；
  ②resume 时若不跳过 seed pretraining，会用 buffer 里仅几百条新随机数据把预训练权重冲掉。

## 失败模式

4M 模型的 4 个失败集（`scripts/fail_diag.py` 输出）集中在两类初始状态：

- **cubeB 贴采样边界**：`|y| ≥ 0.17`（如 `y=+0.295`、`y=-0.186`、`y=-0.171`），机械臂够不到或推动距离过长；
- **两个 cube 初始过近**：`dAB ≈ 0.09`，抓取 A 时先撞飞 B。

统计上 70% 与 80% 的差异不显著（20 集样本的 95% 置信区间约 ±8–9pp），
且只跑了单 seed，因此不能宣称"warm start 带来了提升"。

## 目录结构

```
tdmpc2/                # 算法包（官方代码 + MS3 适配）
scripts/
  install_env.sh       # 环境安装（含 torch 版本约束说明）
  train_2m.sh          # 从零训练 2M
  train_warmstart_4m.sh# warm start 续训 +2M
  eval_20ep.sh         # 20 集确定性终评
  fail_diag.py         # 失败集初始位姿诊断
results/
  ms3_stack_2m/eval.csv
  ms3_stack_4m_ws/eval.csv
docs/report.md         # 完整复现报告（含 MS2 对照、移植决策、失败分析）
```

## 许可

算法代码沿用官方 TD-MPC2 的 MIT License（见 `LICENSE`）。
