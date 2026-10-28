#!/bin/bash
# Stage 1: 2M steps from scratch on ManiSkill3 StackCube-v1 (single seed).
# 与论文口径一致的部分：model_size=5 / batch_size=256 / lr=3e-4 / reward_coef=0.1 / episodic=false
# 与论文口径不同的部分：只在 ManiSkill3 上跑 2M steps（论文是 4M steps × 3 seeds）
set -u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
export VK_ICD_FILENAMES=/root/egl_icd.json   # Vulkan ICD for headless rendering
unset TORCH_LOGS
unset TORCH_TRACE

ROOT=${TDMPC2_ROOT:-/root/tdmpc2}
cd "$ROOT"

rm -rf logs/stack-cube/1/ms3_stack_2m
setsid nohup python tdmpc2/train.py task=stack-cube model_size=5 steps=2000000 \
  episodic=false eval_freq=50000 eval_episodes=10 save_video=true compile=true \
  enable_wandb=false exp_name=ms3_stack_2m seed=1 \
  > /root/ms3_train_2m.log 2>&1 < /dev/null &
sleep 90
echo "=== processes ==="
pgrep -af "train.py" | head -3
echo "=== log head ==="
grep -vE "^$" /root/ms3_train_2m.log | head -12
echo "=== log tail ==="
grep -vE "^$" /root/ms3_train_2m.log | tail -5
df -h / | tail -1
