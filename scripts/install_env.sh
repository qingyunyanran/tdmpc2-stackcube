#!/bin/bash
# Environment for TD-MPC2 + ManiSkill3 StackCube-v1 (tested on AutoDL, single vGPU-32GB).
#
# 关键点：torch 必须降到 2.7.1。torch 2.14 的 dynamo 在 torchrl 的 TensorDict
# 代码路径上会抛 InternalTorchDynamoError: NameError: name 'SYNTHETIC_LOCAL' is not defined
# (torch/_dynamo/guards.py)，一旦 `compile=true` 就必然崩溃；关掉 compile 又只有
# 13-17 agent steps/s。降到 2.7.1 后可以稳定跑 38-39 steps/s。
set -e
source /root/miniconda3/etc/profile.d/conda.sh

echo "=== 1. conda env ==="
conda create -n ms3 python=3.11 -y
conda activate ms3

echo "=== 2. ManiSkill3 ==="
# 实测版本：mani_skill 3.0.1 / sapien 3.0.3 / gymnasium 1.4.0
pip install "mani_skill==3.0.1"

echo "=== 3. torch stack (pinned) ==="
pip install "torch==2.7.1" "tensordict==0.8.3" "torchrl==0.8.1" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== 4. verify ==="
python -c "import torch, tensordict, torchrl, gymnasium, mani_skill, sapien; \
print('torch', torch.__version__); print('tensordict', tensordict.__version__); \
print('torchrl', torchrl.__version__); print('gymnasium', gymnasium.__version__); \
print('mani_skill', mani_skill.__version__); print('sapien', sapien.__version__); \
print('cuda', torch.cuda.is_available())"

cat <<'NOTE'

=== 5. headless rendering ===
所有训练/评估脚本都 export VK_ICD_FILENAMES=/root/egl_icd.json，
它的内容是一个指向 NVIDIA Vulkan ICD 的 json（driver_jsons + library_path）。
没有它 sapien 起不来 renderer，ManiSkill 会在 make() 阶段报 Vulkan 初始化失败。
用 nvidia-smi 找到 libGLX_nvidia.so / nvidia_icd.json 的实际路径后写进该 json 即可。

NOTE
