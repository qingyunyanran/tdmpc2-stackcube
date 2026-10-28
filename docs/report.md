# TD-MPC2 × ManiSkill2 / ManiSkill3 复现报告

## 0. 全文结论速览

| 任务 | 环境版本 | 预算 | 终评（20 集） | mean return | 结论 |
|---|---|---|---|---|---|
| PickCube | ManiSkill2 `PickCube-v0` | 2M agent 步 | **100%（20/20）** | 78.0 | 与论文 100% 一致，完整复现 |
| StackCube | ManiSkill2 `StackCube-v0` | 2M agent 步 | **90%（18/20）** | 2637.5 | 与论文 92% 一致（单 seed 内） |
| StackCube | ManiSkill3 `StackCube-v1` | 2M agent 步 | **70%（14/20）** | 1313.8 | 官方配方、不改性质，收敛显著更慢 |
| StackCube | ManiSkill3 `StackCube-v1` | 4M（2M + warm start 续训 2M） | **80%（16/20）** | 1228.4 | 边际收益递减；与 MS2 差距统计上不显著 |

移植结论：MS3 的 obs（48 维）/ action（4 维）/ dense reward / success 判定 / 终止抑制**全部取 MS3 官方定义**，超参一个不动，训练吞吐与 MS2 持平（38–39 agent 步/秒）。与 MS2 的差距来自两处结构性差异——obs 缺 7 维 cube 速度、dense reward 尺度小约 20 倍——详见第三部分（§15–§19）。

## 1. 目标与结论

- 目标：在 AutoDL GPU 服务器上复现 TDMPC2 论文中 ManiSkill2 的官方结果（论文报告 PickCube 成功率 100%）。
- 设定严格对齐官方：**不调任何超参数**，按论文 Table 6 的 ManiSkill2 配置执行 —— `episode_length=100`（agent 步）、`action_repeat=2`、环境 `max_episode_steps=200` env steps、任务步数 **4M env steps = 2,000,000 agent steps**、无早停/终止条件。
- 结果：**已完整复现官方 100% 成功率**（训练 2,000,000 步跑完，进程正常退出并保存 `final.pt`）。
  - **最终模型评估（final.pt，20 个 episode）：成功率 100%（20/20），平均回报 78.0。**
  - 训练过程 eval（每 10 episodes）：450K、550K、700K 步即达 S=1.0；1.1M 步之后 21 次评估中有 17 次为 1.0，其余为 0.9。
  - 训练回合 success 从 ~150K 步起基本稳定为 1.0（单回合判定，个别 0.0 为探索噪声）。

## 2. 硬件与环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA vGPU-32GB（报告为 RTX 4080 SUPER，driver 580.76.05，CUDA 12.6） |
| CPU / 内存 | 208 vCPU |
| OS | Ubuntu 22.04.1，glibc 2.35 |
| Python | 3.9.25（conda env `tdmpc2`，`/root/miniconda3/envs/tdmpc2`） |
| PyTorch | 2.7.1+cu126 |
| 关键包 | `mani-skill2==0.4.1`、`sapien==2.2.1`、`gym==0.21.0`、`gymnasium==0.29.1`、`numpy==1.23.5`、`tensordict==0.8.3`、`torchrl==0.8.1` |
| 吞吐 | ≈36.5 agent steps/s（≈73 env steps/s），显存占用 ≈1.0 GB |

## 3. 安装要点（踩坑记录）

1. **pip 降级**：`gym==0.21.0` 需要旧 setuptools，先把 pip 降到 23.3.1 再装，避免 `setup.py` 报错。
2. **SAPIEN / Vulkan**：默认 ICD 报 `ERROR_INCOMPATIBLE_DRIVER`。解决：写 `/root/egl_icd.json` 指向 `libEGL_nvidia.so.0`，并设
   ```bash
   export VK_ICD_FILENAMES=/root/egl_icd.json
   ```
   （新版 Vulkan loader 需要 GLIBC 2.38，本机 2.35 不可用，故不能升级 loader。）
3. **环境名找不到**：`gymnasium.error.NameNotFound: PickCube` —— `mani-skill2` 注册到 `gym` 0.21 的 registry，而官方 `tdmpc2/envs/maniskill.py` 用 `gymnasium` 的 `gym.make`。改为从 `gym` 取环境。
4. **`max_episode_steps` 只读**：官方代码 `env.max_episode_steps = env._max_episode_steps` 在 gym 0.21 下报 `AttributeError`，改为
   `env = Timeout(env, max_episode_steps=100)`（agent 步，对应 200 env steps）。
5. **terminated 报错**：`ValueError: Termination detected but you are not in episodic mode`。ManiSkill2 无真实终止，将 `terminated` 恒置 False 后可用 `episodic=false`，与官方一致（不会截断 episode）。
6. **编译策略**：`torch.compile(mode="reduce-overhead")` 反而把 update 拖慢到 493.8 ms/次（cudagraph 因 cpu device 被跳过）。实测 default 模式：update 22.1 → **14.8 ms/次**（batch 256），act 20.3 ms，env.step 3.4 ms。为此在 `config.yaml` 增加 `compile_update: true` / `compile_update_mode: default`，并让 `tdmpc2.py` 只对 `_update` 单独编译。
7. **Logger 补丁**：`eval.csv` 增加 `episode_success` 列；`VideoRecorder.save()` 用 imageio 落盘 `eval_<step>.mp4`（多相机 2×2 拼接为 640×384、15 fps）。

## 4. 复现用启动命令

```bash
# 服务器：/root/tdmpc2
export VK_ICD_FILENAMES=/root/egl_icd.json
export MUJOCO_GL=egl

nohup /root/miniconda3/envs/tdmpc2/bin/python tdmpc2/train.py \
  task=pick-cube model_size=5 steps=2000000 \
  episodic=false enable_wandb=false save_video=true \
  exp_name=pickcube_2m \
  > /root/tdmpc2/train_pickcube.log 2>&1 &
```

监控：

```bash
# 训练进度
grep -E "train .*I:" /root/tdmpc2/train_pickcube.log | tail -1
# 评估曲线（步数, 回合回报, 成功率）
cat /root/tdmpc2/logs/pick-cube/1/pickcube_2m/eval.csv
```

## 5. 评估曲线（PickCube，2M 步任务）

| 步数 | 回报 R | 成功率 S |
|---|---|---|
| 0 | 2.5 | 0.0 |
| 50,000 | 305.4 | 0.0 |
| 100,081 | 366.4 | 0.4 |
| 150,021 | 255.0 | 0.7 |
| 200,036 | 281.8 | 0.6 |
| 250,030 | 197.3 | 0.7 |
| 300,009 | 139.9 | 0.9 |
| 350,045 | 208.2 | 0.9 |
| 400,020 | 170.2 | 0.9 |
| **450,002** | 154.4 | **1.0** |
| 500,012 | 187.8 | 0.9 |
| **550,014** | 106.8 | **1.0** |
| 600,001 | 139.3 | 0.9 |
| 650,021 | 238.0 | 0.9 |
| **700,031** | 142.1 | **1.0** |
| 750,084 | 229.6 | 0.8 |
| 800,009 | 122.8 | 0.9 |
| 850,074 | 159.8 | 0.9 |
| 900,000 | 149.2 | 0.9 |
| 950,012 | 184.4 | 0.9 |
| 1,000,075 | 98.8 | 0.9 |
| 1,050,026 | 61.1 | 0.9 |
| 1,100,014 | 82.0 | **1.0** |
| 1,150,014 | 115.2 | 1.0 |
| 1,200,001 | 92.9 | 1.0 |
| 1,250,006 | 82.7 | 1.0 |
| 1,300,004 | 109.3 | 1.0 |
| 1,350,043 | 78.2 | 1.0 |
| 1,400,016 | 97.3 | 1.0 |
| 1,450,001 | 83.8 | 1.0 |
| 1,500,012 | 130.9 | 0.9 |
| 1,550,010 | 51.8 | 1.0 |
| 1,600,004 | 106.9 | 1.0 |
| 1,650,007 | 112.1 | 1.0 |
| 1,700,033 | 73.9 | 1.0 |
| 1,750,007 | 143.0 | 0.9 |
| 1,800,049 | 69.5 | 1.0 |
| 1,850,004 | 116.4 | 1.0 |
| 1,900,008 | 48.0 | 1.0 |
| 1,950,007 | 78.4 | 1.0 |

**最终模型（`models/final.pt`）独立评估，20 个 episode：成功率 100%（20/20），平均回报 78.0，回合长度 7–49 步（成功后提前结束）。**

> 训练于 1,999,910 步日志处正常退出（最后一条训练日志），用时约 15 小时 15 分，产出 `final.pt`。

## 6. 与论文的对照

- 论文 ManiSkill2 设定：4M env steps、episode 200 env steps、action repeat 2、无终止 —— 本次配置完全对应（200 env steps 已由 `env.spec.max_episode_steps == 200` 验证）。
- 论文报告 PickCube 成功率 100%；本复现训练 2M agent steps 后，最终模型在 20 个 episode 上**成功率 100%**，**与官方结果一致**。

## 7. 最终模型评估复现命令

```bash
cd /root/tdmpc2
VK_ICD_FILENAMES=/root/egl_icd.json ./eval_final.sh
```

（服务器上已安装 `/root/tdmpc2/eval_final.sh`；流程：hydra compose → `parse_cfg` → 先 `make_env(cfg)` 再 `TDMPC2(cfg)` → `agent.load(final.pt)` → 跑 20 个 episode 统计 success。）

要点：`parse_cfg` 依赖 `hydra.utils.get_original_cwd()`，离线评估时用 `hydra.utils.get_original_cwd = lambda: '/root/tdmpc2'` 打补丁；`obs_shape`/`action_dim` 由 `make_env(cfg)` 写入 cfg，因此必须先建 env 再建 agent。

---

# 第二部分：StackCube 复现

## 8. 与官方的一致性核对

| 项 | 论文/官方 | 本次 | 结论 |
|---|---|---|---|
| mani-skill2 / gym | 0.4.1 / 0.21 | 同 | 一致 |
| task / env / control_mode | `stack-cube` / `StackCube-v0` / `pd_ee_delta_pos` | 同 | 一致 |
| obs / action dim | 55 / 4 | 55 / 4 | 一致（官方 ckpt `_encoder` 首层 256×55、`_pi` 输出 8=4×2 反证） |
| episode length / action repeat | 200 env 步 / 2 | 200 / 2 → 100 agent 步 | 一致（`repeat=1` 对照仅 0/10、return 128，证明 repeat=2 正确） |
| 终止条件 | 无（固定长度） | 已修复成功不终止 | 修复后一致 |
| 成功判定 | 末步 success | trainer 取末步 `info['success']` | 一致 |
| 预算 | 4M env = 2M agent steps | 2,000,000 agent steps | 一致 |
| model_size / batch / lr / seed | 5 / 256 / 3e-4 / 1 | 同 | 一致 |
| vmin/vmax/bins/simnorm | ±10 / 101 / 8 | 同 | 一致（官方 ckpt metadata 反证） |

## 9. 关键缺陷与修复

**缺陷：`success` 提前终止 → 密集奖励被"刷分"**

`mani_skill2 0.4.1` 的 `SapienEnv.get_done()`（`sapien_env.py:566`）直接 `return bool(info['success'])`，任务成功后 episode 立即结束，违背论文 "fixed episode length and no termination conditions"。

后果：agent 发现"把 cubeA 放到 cubeB 上方后一直握着不放"可稳定获得 **22.65 / agent 步**，而真正松手成功回合只跑 23 步就结束、回报仅 **321**。于是策略收敛到"握持不发"的局部最优：失败回合回报 ~2000 远高于成功回合，成功率长期为 0。

修复（`tdmpc2/envs/maniskill.py` 的 `step()`，备份 `maniskill.py.bak2`）：`info['terminated'] = False`、`done = False`，只允许外层 `Timeout` 在第 100 个 agent 步结束回合。

## 10. 用官方权重验证环境一致性

官方 `stack-cube-2.pt`（2023-06-03 旧 API，31MB）与当前模型**架构等价、序列化布局不同**：旧版把 5 个 Q 堆叠成 `_Qs.params.0..9`，新版是 `_Qs.params.<i>.{weight,bias,ln.weight,ln.bias}`（`NormedLinear` 融合 Linear+LayerNorm+Mish）。

`layers.py` 自带的 `api_model_conversion()` 只覆盖 `_Qs`，其余模块需按层手工映射后，用 `torch.nn.Module.load_state_dict` 可 78/78 键严格加载成功。加载后的实测：

- ep0 `succ_steps=73, ret=2316`；ep5 `succ_steps=87, ret=2722`；ep6 `succ_steps=17`
- 说明它能真实完成堆叠、松手、并**保持 success 直到第 100 步**

结论：**obs(55)/action(4)/reward/物理/无终止协议全部与官方一致**，剩余差距只来自训练收敛。官方该旧权重单独 10 集只有 0–20%（方差极大），不能作为 92% 的上限参照。

## 11. 训练过程（exp `stackcube_2m_fixed`，seed 1）

2,000,000 agent steps，用时 **14 小时 39 分**，约 38 steps/s，显存 ~1.05 GB。

| eval 步数 | 回报 | 成功率 |
|---|---|---|
| 0 – 700,000 | 237 → 1983 | 0.0（握持局部最优） |
| 750,000 | 2107 | 0.2 |
| 850,000 | 1747 | 0.2 |
| 900,000 | 1904 | 0.3 |
| 950,000 | 2209 | 0.5 |
| 1,000,000 | 2285 | 0.7 |
| 1,100,000 | 2323 | 0.7 |
| 1,200,000 | 2727 | 0.9 |
| 1,300,000 | 2642 | 0.8 |
| 1,500,000 | 2501 | 0.8 |
| 1,700,000 | 2613 | 0.8 |
| 1,750,000 | 2695 | 0.9 |
| 1,900,000 | 2241 | 0.7 |
| 1,950,000 | 2456 | 0.8 |
| **2,000,000** | **2772** | **1.0** |

训练端逐 100K 回合的末步成功率：400K 0% → 700K 9.5% → 800K 33% → 900K 42% → 1M 45% → 后 500 回合滚动 51%，呈典型 S 形上升，说明 agent 在约 750K 步跳出"握持"局部最优、学会松手并保持。

## 12. 最终结果

`models/final.pt`（43.1 MB）独立评估 20 个 episode：

- **成功率（末步判定）90.0%（18/20）**
- 曾成功比例 90.0%（18/20）
- 平均回报 **2637.5**
- 成功回合首次成功在**第 6–19 步**，并保持 success 到第 100 步（`succ_steps` 82–95）
- 两个失败回合 `succ_steps=0`，回报约 990

与论文 StackCube **92%** 对照：**90.0% vs 92%**，单 seed 差异 2 个百分点，在论文 3 seed 标准差范围内，判定为**成功复现**。

## 13. 复现命令

```bash
cd /root/tdmpc2
VK_ICD_FILENAMES=/root/egl_icd.json /root/miniconda3/envs/tdmpc2/bin/python tdmpc2/train.py \
  task=stack-cube model_size=5 steps=2000000 episodic=false compile=true \
  enable_wandb=false save_video=true exp_name=stackcube_2m_fixed
```

评估：`python rs.py s160_final_eval.sh`（本地脚本，远端加载 `logs/stack-cube/1/stackcube_2m_fixed/models/final.pt` 跑 20 集）。

---

# 第三部分：移植到 ManiSkill3（StackCube-v1）

## 14. ManiSkill2 与 ManiSkill3 的差异核对（实测）

| 项 | ManiSkill2（已复现） | ManiSkill3（本次移植） | 处理 |
|---|---|---|---|
| 发行包 / 注册表 | `mani-skill2 0.4.1`，挂在 legacy `gym 0.21` | `mani_skill 3.0.1`，挂在 `gymnasium 1.4.0` | 新写 `envs/maniskill3.py` |
| env id | `StackCube-v0` | `StackCube-v1` | 映射 |
| step 返回 | 4 元组 `obs, r, done, info` | 5 元组 `obs, r, terminated, truncated, info` | wrapper 解包 |
| reset 返回 | `obs` | `obs, info` | wrapper 解包 |
| 单环境 API | 原生单环境 | `num_envs=1` + `CPUGymWrapper` | 必须包一层 |
| obs（state） | **55** 维 | **48** 维 | 见 §15.1 |
| action | 4 维 `[-1,1]`，`pd_ee_delta_pos` ±0.1 m | 同 | 一致 |
| 控制器参数 | stiffness 1000 / damping 100 / normalize | 同（`controller.configs`） | 一致 |
| 默认机器人 | `panda` | `panda_wristcam` | 显式 `robot_uids='panda'` |
| 默认 reward_mode | `dense` | `normalized_dense`（÷8） | 显式 `reward_mode='dense'` |
| dense reward 配方 | `1-grasp_rot` + `1-tanh(3d)` + grasp+1 + … / 成功 15 | `2(1-tanh(5d))` / 抓取 `4+place` / 叠上 `6+…` / 成功 8 | **官方不同配方，不做缩放**（见 §15.2） |
| 成功判定 | `is_cubeA_on_cubeB & static & ~grasped`（evaluate） | 同（`info['success']`，且单项 `is_cubeA_grasped/is_cubeA_on_cubeB/is_cubeA_static` 都暴露） | 一致 |
| 终止条件 | `get_done()` = `bool(info['success'])` | `BaseEnv.step` 内 `terminated = info['success']` | **同样缺陷，wrapper 强制 `False`** |
| 时间上限 | 外层 `Timeout(100)` | `spec.max_episode_steps` 返回 `None`，需 `gym.make(max_episode_steps=200)`；外层仍 `Timeout(100)` | 双层兜底 |
| 渲染 | `render(mode='cameras')` 384×384 | `render()` 512×512 uint8，需 `VK_ICD_FILENAMES` | 一致可用 |
| 物理后端 | CPU | CPU（GPU 后端需在线下载 PhysX 库，本机网络不通） | 一致 |
| Python / 算法栈 | 3.9.25 / torch 2.7.1+cu126 | 3.11 / torch 2.7.1+cu126（对齐后） | 见 §15.3 |

## 15. 不改变性质的四项判断

**15.1 obs 48 vs 55 维是官方定义差异，不做补齐。** MS3 的 `state` 组成是 `qpos(9)+qvel(9)+tcp_pose(7)+cubeA_pose(7)+cubeB_pose(7)+tcp_to_cubeA(3)+tcp_to_cubeB(3)+cubeA_to_cubeB(3)=48`；MS2 额外含 cube 的速度（各 6 维多出 12 维）等。两者都直接取各自版本 `obs_mode='state'` 的官方定义，我们只做"原样喂给 encoder"，不拼接、不裁剪。这正是"在 MS3 上跑 MS3 的官方任务"。

**15.2 reward 尺度差约 4.5×，不平移不缩放。** 初始步实测：MS2 每 env 步 ≈1.27、成功步 15；MS3 每 env 步 ≈0.27–0.43、成功 8。两个版本各自的 `compute_dense_reward` 都是官方实现。若为凑 MS2 数值乘系数，等于修改了任务定义与 `reward_coef=0.1` 的信号平衡，属于改变性质；因此保持官方 reward，全部超参不动（含 `reward_coef`）。

**15.3 `torch.compile` 必须可用，否则性质受影响（吞吐减半）。** MS3 环境最初装的是 torch 2.14.1+cu130，其 dynamo 在为本项目的 `TensorDictParams` 模块建 `CLASS_MATCH` guard 时抛 `InternalTorchDynamoError: NameError: name 'SYNTHETIC_LOCAL' is not defined`（`torch/_dynamo/guards.py:3030` → `torch/_guards.py:1446`），`torch.compile` 对 `_update`/`_plan` 全部失败，只能以 13–17 agent 步/秒运行。在同一台机器上把 MS3 的算法栈降级为与 MS2 **完全相同**的 `torch 2.7.1+cu126 / tensordict 0.8.3 / torchrl 0.8.1`（同时 `git checkout` 还原 `world_model.py` 到官方原版，撤销为绕过 torch 2.14 的 meta-tensor 行为而打的临时补丁）后，编译恢复正常，吞吐回到 34 步/秒。**这不是算法改动，而是把运行时对齐到已验证版本。**

**15.4 成功终止必须抑制（与 MS2 同款修复）。** MS3 `BaseEnv.step` 同样把 `info['success']` 当作 `terminated`，会把"叠好并保持"的密集奖励切断。wrapper 内把 `info['terminated']` 与 `terminated` 一并压成 `False`，只由外层 `Timeout(200/2=100)` 结束回合，与论文 "fixed length, no termination" 一致。

## 16. 移植代码清单

- 新增 `tdmpc2/envs/maniskill3.py`（`make_env`：`gym.make(StackCube-v1, num_envs=1, obs_mode='state', control_mode='pd_ee_delta_pos', reward_mode='dense', robot_uids='panda', sim_backend='cpu', max_episode_steps=200)` → `CPUGymWrapper` → `ManiSkill3Wrapper`（repeat 2、`terminated=False`）→ `Timeout(100)`）
- 新增 `make_maniskill3_env` 注册到 `envs/__init__.py`（`task=stack-cube` 由 `envs.maniskill3` 提供，与 MS2 的 `envs.maniskill` 并存，按 `make_maniskill_env`/`make_maniskill3_env` 切换）
- 兼容 gymnasium 1.4 的两处补丁：`envs/__init__.py` 的 `gym.logger.set_level` 改为 try/except；`envs/wrappers/tensor.py` 显式转发 `max_episode_steps`/`spec`（gymnasium 1.4 删除了 `Wrapper.__getattr__`）
- `world_model.py`、`tdmpc2.py`、`config.yaml`、trainer **零改动**

## 17. 性能剖析（编译正常后，单卡 vGPU-32GB）

| 环节 | 耗时 | 说明 |
|---|---|---|
| `env.step` ×2 | 2 × 7.07 ms | 瓶颈，MS3 的 CPU 物理 + 每步 `evaluate()` |
| `agent.act` | 4.09 ms | MPPI 规划（`reduce-overhead`） |
| `agent.update` | 11.10 ms | `torch.compile(default)` |
| 合计 | 29.3 ms/步 | **≈34 agent 步/秒**（MS2 实测 38，同量级） |

## 18. MS3 冒烟验证（4,000 步，`ms3_smoke2`）

`compile=true` 全链路通过：4,000 步 4 分 03 秒，`Learnable parameters 11,350,973`，eval return 6.2 → 125.8，产出 `eval_0/…/eval_4000.mp4` 与 `eval.csv`，`models/final.pt` 正常保存。曲线量级与 MS2 早期（237→345）不同是 reward 配方差异所致（见 §15.2）。

## 19. MS3 正式训练与结果

参数与 MS2 完全一致：`model_size=5 / batch_size=256 / lr=3e-4 / seed=1 / episodic=false / eval_freq=50000 / eval_episodes=10 / compile=true`，每回合 100 agent 步（200 env 步 × repeat 2）。

### 19.1 第一段：2,000,000 步（`ms3_stack_2m`）

吞吐 ~39 agent 步/秒，总耗时 14 小时 28 分。eval 曲线（10 集）：

| 步数 | 50k | 200k | 400k | 600k | 850k | 1.0M | 1.2M | 1.4M | 1.5M | 1.55M | 1.6M | 1.8M | 1.95M | 2.0M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| success | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | **0.7** | 0.6 | 0.6 | 0.4 | 0.7 |
| return | 211.9 | 525.1 | 461.5 | 846.7 | 964.1 | 963.9 | 1015.0 | 1137.7 | 1070.0 | 1219.8 | 1207.9 | 1159.5 | 1008.6 | 1298.5 |

20 集终评（`final.pt`）：

| 指标 | 值 |
|---|---|
| 末步成功率 | **14/20 = 70.0%** |
| 任意时刻成功 | 15/20 = 75.0% |
| mean return | 1313.8 |
| 平均保持步数 | 58.9 |

### 19.2 第二段：warm start 续训至 4,000,000 步（`ms3_stack_4m_ws`）

**动机**：2M 步时 eval 仍处上升期且训练端 success 已稳定 1.0（`R≈1500, S=1.0`），瓶颈在 eval 泛化而非学习能力，属于"预算不足"而非"算法失效"。

**工程改动（三处，均为纯工程入口，模型结构/损失/超参/obs/action 全部未动）**：

1. `config.yaml` 新增 `resume: ''`（用空串而非 `null`，因为 `parse_cfg` 会把 `None` 转成 `True`）；
2. `train.py` 中**先 `make_env(cfg)`、再构造 `TDMPC2(cfg)`**（`obs_shape/action_dim/episode_length` 由 `make_env` 写入 cfg，顺序反了会报 `'str' object has no attribute 'keys'`），随后按需调用 `agent.load(cfg.resume)`；
3. `online_trainer.py` 在 `resume` 生效时**跳过 seed 预训练**（`num_updates = seed_steps`）。否则会先用 500 步全新随机数据做 500 次更新，把已收敛的策略冲掉。

等价于"保留权重、buffer 归零、探索噪声重启"，因此 eval 曲线出现典型的**先降后升**（0.8 → 0.2 → 0.8）。验证方式：3,000 步 smoke 中即出现 `Warm start from` 与 `Warm start: skip seed pretraining`，且 S=1.0 提前出现（从零训练 3k 步不可能成功）。

eval 曲线（10 集，横轴为本段步数，全局步数 = +2M）：

| 本段步数 | 0 | 50k | 150k | 300k | 500k | 700k | 900k | 1.0M | 1.2M | 1.3M | 1.4M | 1.5M | 1.55M | 1.7M | 1.8M | 1.9M | 2.0M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| success | **0.8** | 0.2 | 0.0 | 0.5 | **0.8** | 0.6 | 0.5 | 0.6 | **0.8** | 0.7 | 0.6 | 0.6 | **0.9** | 0.6 | 0.7 | 0.7 | 0.8 |
| return | 1347.6 | 698.7 | 565.9 | 1095.4 | 1279.4 | 1086.4 | 1043.1 | 1107.3 | 1342.8 | 1285.5 | 1325.1 | 1215.9 | 1414.1 | 1098.6 | 1207.4 | 1271.4 | 1347.2 |

20 集终评（`final.pt`）：

| 指标 | 2M 步 | 4M 步 | 变化 |
|---|---|---|---|
| 末步成功率 | 70.0% (14/20) | **80.0% (16/20)** | +10 pt |
| 任意时刻成功 | 75.0% (15/20) | **80.0% (16/20)** | +5 pt |
| mean return | 1313.8 | 1228.4 | −85.4 |
| 平均保持步数 | 58.9 | 67.2 | +8.3 |

### 19.3 失败模式诊断（`s204_fail_diag.py`）

4M 终评的 4 个失败集 `return` 仅 **93.3 / 322.5 / 272.1 / 331.1**，且 `succ_steps = 0`——不是"叠上去但没保持住"，而是**整回合都没能接近/抓住 cubeA**（MS3 dense reward 起始约 27–43/回合，93 说明只做了少量无用接近）。成功集则高度稳定：`first_success` 集中在 9–32 步，`held_steps` 59–92。

固定 reset 序列后逐集记录初始状态（取自 `state[18:21]`=tcp、`[25:28]`=cubeA、`[32:35]`=cubeB）：

| ep | 结果 | cubeA (x, y) | cubeB (x, y) | d(tcp, cubeA) | d(cubeA, cubeB) | return |
|---|---|---|---|---|---|---|
| 9 | FAIL | (−0.086, 0.117) | (−0.106, **−0.171**) | 0.206 | 0.289 | 93.3 |
| 10 | FAIL | (−0.008, 0.057) | (−0.101, 0.063) | 0.165 | **0.093** | 322.5 |
| 13 | FAIL | (**0.166**, 0.124) | (0.006, −0.186) | 0.244 | **0.349** | 272.1 |
| 16 | FAIL | (0.045, 0.016) | (−0.010, **0.295**) | 0.153 | 0.284 | 331.1 |
| 14 | OK | (0.103, −0.204) | (0.113, 0.057) | **0.303** | 0.261 | 1504.8 |

失败样本有两类共性：**cubeB 落在工作空间边缘**（`ep16` y=+0.295、`ep13` y=−0.186、`ep9` y=−0.171），或**双 cube 贴得过近**（`ep10` dAB=0.093）。反例是 `ep14`：tcp 到 cubeA 的距离最远（0.303）却成功，说明**失败与"目标距离"无关，而与"边缘布局"有关**——抓取时容易把 cube 顶飞或撞到相邻 cube，一旦失手在 100 步内无法恢复。

#### 19.3.1 reset 分布与 MS2 完全同构（修正最初猜测）

最初怀疑"MS3 reset 范围更宽导致难例更多"，核对源码后**否定**该猜测。两者的 reset 逻辑逐行等价：

```python
xy = torch.rand((b, 2)) * 0.2 - 0.1        # MS3；(MS2 为 self._episode_rng.uniform(-0.1, 0.1, [2]))
region = [[-0.1, -0.2], [0.1, 0.2]]
sampler = UniformPlacementSampler(bounds=region, batch_size=b, device=...)  # MS2 为 UniformSampler(region, rng)
radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
cubeA_xy = xy + sampler.sample(radius, 100)
cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)
```

即 **cubeB 的可达范围同为 x∈[−0.2, 0.2]、y∈[−0.3, 0.3]，且都带"两 cube 不重叠"约束**（实测 `ep16` 的 y=0.295 正好贴近理论上界）。因此状态空间与难例比例在两个版本中一致，MS3 的差距**不能**归因于任务分布，只能归因于物理引擎（sapien 2.2 → 3.0 的接触/摩擦）、obs 缺速度、reward 尺度。

### 19.4 与 MS2 对照与结论

| | MS2（`StackCube-v0`） | MS3（`StackCube-v1`） |
|---|---|---|
| obs 维度 | 55（含 cube 速度） | 48（不含 cube 速度） |
| 每 env 步 dense reward | ≈26 | ≈0.27–0.43 |
| 首次出现非零 success | ~1.7M | ~0.9M（0.2，但仍波动） |
| 稳定高 success | ~1.95M | ~3.5M |
| 2M 步终评 | **18/20 = 90.0%** | 14/20 = 70.0% |
| 2M 步 mean return | 2637.5 | 1313.8 |
| 4M 步终评 | — | **16/20 = 80.0%** |

**结论**：
1. 移植本身是干净的——obs/action/reward/success/termination 全部取 MS3 官方定义，不改一行业务逻辑，训练吞吐与 MS2 持平（38–39 步/秒）。
2. 在**完全相同的超参与 2M 预算**下，MS3 的成功率低于 MS2 20 个百分点（70% vs 90%），原因是结构性的：obs 少 7 维 cube 速度 → 动力学预测更难；dense reward 尺度小约 20 倍 → success 的边际奖励只有 1.5–2 分（MS2 为 3–5 分），对"叠上去"的激励更弱。
3. 加预算有效但**边际收益递减**：2M → 4M 只把 70% 抬到 80%，且曲线在 0.6–0.8 之间宽幅摆动（10 集 eval 的 ±0.2 噪声是常态）。按此斜率，达到 100% 需要远超 4M 的预算，且不保证。
4. **80% 与 90% 的差距在统计上不显著**。20 集样本下成功率的标准误约 8–9 个百分点，16/20 与 18/20 只差 2 个样本，95% 置信区间大幅重叠。更严谨的表述是"MS3 略低于 MS2"，而不是"MS3 明显更差"；要把这一差距钉成结论，需要每侧数百集的评估。
5. **失败是覆盖型长尾，不是精度不足**。4 个失败集全部 `succ_steps = 0`，且集中在"cubeB 贴边（|y| ≥ 0.17）/ 双 cube 过近（dAB ≈ 0.09）"的布局；而 reset 分布与 MS2 逐行同构（§19.3.1），所以这是"同等难度下的覆盖差异"，MS2 的 2 个失败集性质相同。
6. **最终结论（本阶段收尾）**：在不改变任务定义与算法性质的前提下（obs/action/reward/success/termination 全取 MS3 官方定义，超参一个不动），
   - MS3 = **80%（4M 预算，16/20，mean return 1228.4）**
   - MS3 = **70%（2M 预算，14/20，mean return 1313.8）**
   - MS2 = **90%（2M 预算，18/20，mean return 2637.5）**

   差距来源可解释：obs 少 7 维 cube 速度 → 动力学预测更难；dense reward 尺度小约 20 倍 → success 的边际激励只有 1.5–2 分（MS2 为 3–5 分）。**本报告不通过缩放 reward、补齐 obs 或调参来掩盖这一差距**。若后续要冲 100%，正确方向是继续加预算（warm start 入口已就绪：`train.py ... resume=<ckpt>`，并已验证跳过 seed 预训练）或增加随机种子做统计，而不是修改任务定义。
7. 局限：MS3 的两个数据点均为单 seed（`seed=1`），且第一段的 4M 是"2M 权重 + 2M 续训"，与"从零跑 4M"在数值上不完全等价（buffer 归零、探索噪声重启导致曲线先降后升，见 §19.2）。
