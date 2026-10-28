#!/bin/bash
# Stage 2: warm start from the 2M checkpoint and run another 2M steps (total 4M).
#
# 两个必要前提（已包含在 tdmpc2/ 代码里，无需再打补丁）：
#   1) config.yaml 里的 `resume: ''` + train.py 中 `agent.load(cfg.resume)`
#   2) online_trainer.py 在 resume 非空时跳过 seed pretraining
#      （否则会用 buffer 里仅有的几百条新随机数据做 seed_steps 次更新，把预训练权重冲掉）
#
# 已知坑：必须 `make_env(cfg)` 先于 `TDMPC2(cfg)`，
# 因为 TDMPC2 构造需要 cfg.obs_shape / cfg.action_dim / cfg.episode_length，
# 这些值由 make_env 写入 cfg。顺序反了会报
#   AttributeError: 'str' object has no attribute 'keys'
set -u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
unset TORCH_LOGS
unset TORCH_TRACE
export VK_ICD_FILENAMES=/root/egl_icd.json

ROOT=${TDMPC2_ROOT:-/root/tdmpc2}
cd "$ROOT"

CKPT=${CKPT:-$ROOT/logs/stack-cube/1/ms3_stack_2m/models/final.pt}
if [ ! -f "$CKPT" ]; then echo "MISSING_CKPT $CKPT"; exit 1; fi

echo "=== smoke: 3000 steps warm start (check resume + skip-seed) ==="
python tdmpc2/train.py task=stack-cube model_size=5 steps=3000 episodic=false \
  eval_freq=1000000 eval_episodes=1 save_video=false compile=true enable_wandb=false \
  resume=$CKPT exp_name=ms3_ws_smoke seed=1 > /root/ms3_ws_smoke.log 2>&1
echo "exit=$?"
grep -c "Warm start from" /root/ms3_ws_smoke.log
grep -c "Warm start: skip seed pretraining" /root/ms3_ws_smoke.log
grep -vE "^$" /root/ms3_ws_smoke.log | tail -6

if [ "$(grep -c 'Warm start from' /root/ms3_ws_smoke.log)" -ge 1 ] && \
   [ "$(grep -c 'Warm start: skip seed pretraining' /root/ms3_ws_smoke.log)" -ge 1 ]; then
  if pgrep -f "exp_name=ms3_stack_4m_ws" > /dev/null; then
    echo "ALREADY_RUNNING"
  else
    nohup python tdmpc2/train.py task=stack-cube model_size=5 steps=2000000 episodic=false \
      eval_freq=50000 eval_episodes=10 save_video=true compile=true enable_wandb=false \
      resume=$CKPT exp_name=ms3_stack_4m_ws seed=1 > /root/ms3_train_4m.log 2>&1 &
    sleep 20
    echo "LAUNCHED"; pgrep -af "exp_name=ms3_stack_4m_ws" | head -1
  fi
else
  echo "SMOKE_FAILED"
fi
df -h / | tail -1
