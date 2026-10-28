#!/bin/bash
# 20-episode final evaluation (deterministic policy, no exploration noise).
#
#   CKPT=<path to final.pt> EPISODES=20 bash scripts/eval_20ep.sh
#
# 报告两个指标：
#   FINAL_SUCCESS_STEP  最后一步是否仍处于 success（论文口径）
#   FINAL_SUCCESS_ANY   该集是否曾经成功过（更宽松）
set -u
ROOT=${TDMPC2_ROOT:-/root/tdmpc2}
cd "$ROOT"

export CKPT="${CKPT:-$ROOT/logs/stack-cube/1/ms3_stack_2m/models/final.pt}"
export EPISODES="${EPISODES:-20}"
export TDMPC2_ROOT="$ROOT"

MS_BACKEND=ms3 VK_ICD_FILENAMES=/root/egl_icd.json /root/miniconda3/envs/ms3/bin/python - <<'EOF' 2>&1 | grep -vE "svulkan2|GLFW|DISPLAY" | tail -40
import os, sys, torch, numpy as np, hydra.utils
os.environ['MS_BACKEND'] = 'ms3'
ROOT = os.environ['TDMPC2_ROOT']
sys.path.insert(0, os.path.join(ROOT, 'tdmpc2'))
hydra.utils.get_original_cwd = lambda: ROOT
from hydra import compose, initialize
from common.parser import parse_cfg

CKPT = os.environ['CKPT']
EPISODES = int(os.environ.get('EPISODES', '20'))
with initialize(config_path='tdmpc2', version_base=None):
    cfg = parse_cfg(compose(config_name='config', overrides=[
        'task=stack-cube', 'model_size=5', 'steps=2000000', 'episodic=false',
        'compile=false', 'enable_wandb=false', 'save_video=false', 'exp_name=final_eval_ms3']))
cfg.device = 'cuda'
from tdmpc2 import TDMPC2
from envs import make_env

def succ_of(info):
    v = info.get('success', False)
    try:
        return bool(np.asarray(v).ravel()[0])
    except Exception:
        return bool(v)

torch.manual_seed(1)
env = make_env(cfg)
print('obs_shape', cfg.obs_shape, 'action_dim', cfg.action_dim, 'episode_length', cfg.episode_length)
agent = TDMPC2(cfg)
agent.load(CKPT)
agent.model.eval()
print('loaded', CKPT, '| episodes', EPISODES)

STEP, ANY, RET, HELD = [], [], [], []
for ep in range(EPISODES):
    obs = env.reset(); t = 0; ret = 0.0; succ = 0; first = None; done = False; info = {}
    while not done and t < 100:
        a = agent.act(obs, t0=(t == 0), eval_mode=True)
        obs, r, done, info = env.step(a)
        ret += r; t += 1
        if succ_of(info):
            succ += 1
            if first is None:
                first = t
    fin = succ_of(info)
    STEP.append(fin); ANY.append(succ > 0); RET.append(ret); HELD.append(succ)
    print('ep%2d len=%3d ret=%7.1f succ_steps=%2d first=%-4s final=%s' % (ep, t, ret, succ, first, fin))
print('FINAL_SUCCESS_STEP = %.3f (%d/%d)' % (np.mean(STEP), sum(STEP), EPISODES))
print('FINAL_SUCCESS_ANY  = %.3f (%d/%d)' % (np.mean(ANY), sum(ANY), EPISODES))
print('FINAL_RETURN       = %.1f' % np.mean(RET))
print('FINAL_HELD_STEPS   = %.1f' % np.mean(HELD))
EOF
