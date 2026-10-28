"""Diagnose which initial states the policy fails on.

For each episode it records the initial tcp / cubeA / cubeB positions read straight
out of the 48-dim observation vector, so failed rollouts can be traced back to
unlucky resets (cube pushed to the edge of the 0.2 x 0.4 sampling region, or the
two cubes starting almost on top of each other).

Observation layout (ManiSkill3 StackCube-v1, 48 dims):
    [ 0: 9] qpos
    [ 9:18] qvel
    [18:25] tcp_pose      (xyz at 18:21)
    [25:32] cubeA_pose    (xyz at 25:28)
    [32:39] cubeB_pose    (xyz at 32:35)
    [39:42] tcp_to_cubeA
    [42:45] tcp_to_cubeB
    [45:48] cubeA_to_cubeB

Usage:
    CKPT=/path/to/final.pt python scripts/fail_diag.py
"""
import os, sys, torch, numpy as np, hydra.utils

os.environ['MS_BACKEND'] = 'ms3'
os.environ.setdefault('VK_ICD_FILENAMES', '/root/egl_icd.json')

ROOT = os.environ.get('TDMPC2_ROOT', '/root/tdmpc2')
sys.path.insert(0, os.path.join(ROOT, 'tdmpc2'))
hydra.utils.get_original_cwd = lambda: ROOT
# hydra 的 config_path 是相对当前工作目录解析的，必须先切到仓库根目录
os.chdir(ROOT)

from hydra import compose, initialize_config_dir
from common.parser import parse_cfg

CKPT = os.environ.get('CKPT', os.path.join(ROOT, 'logs/stack-cube/1/ms3_stack_4m_ws/models/final.pt'))
EPISODES = int(os.environ.get('EPISODES', '20'))

# initialize(config_path=...) 是相对脚本文件解析的，只在 cwd == 仓库根时好用；
# 改用绝对 config 目录后可从任意目录运行
with initialize_config_dir(config_dir=os.path.join(ROOT, 'tdmpc2'), version_base=None):
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
agent = TDMPC2(cfg)
agent.load(CKPT)
agent.model.eval()
print('loaded', CKPT)

rows = []
for ep in range(EPISODES):
    obs = env.reset()
    o = obs['state'] if isinstance(obs, dict) else obs
    s0 = np.asarray(o, dtype=np.float64).ravel()
    tcp = s0[18:21]
    cubeA = s0[25:28]
    cubeB = s0[32:35]
    d_tcp_A = float(np.linalg.norm(tcp - cubeA))
    d_A_B = float(np.linalg.norm(cubeA - cubeB))
    t = 0; ret = 0.0; succ = 0; done = False; info = {}
    while not done and t < 100:
        a = agent.act(obs, t0=(t == 0), eval_mode=True)
        obs, r, done, info = env.step(a)
        ret += r; t += 1
        if succ_of(info):
            succ += 1
    fin = succ_of(info)
    rows.append((ep, tcp.copy(), cubeA.copy(), cubeB.copy(), d_tcp_A, d_A_B, ret, succ, fin))
    print('ep%2d %s tcp=[%.3f %.3f %.3f] A=[%.3f %.3f %.3f] B=[%.3f %.3f %.3f] dTA=%.3f dAB=%.3f ret=%6.1f succ_steps=%2d final=%s'
          % (ep, 'OK  ' if fin else 'FAIL', tcp[0], tcp[1], tcp[2], cubeA[0], cubeA[1], cubeA[2],
             cubeB[0], cubeB[1], cubeB[2], d_tcp_A, d_A_B, ret, succ, fin))

ok = [r for r in rows if r[8]]
bad = [r for r in rows if not r[8]]
print('\n=== summary ===')
print('success %d/%d' % (len(ok), len(rows)))
for name, grp in (('OK', ok), ('FAIL', bad)):
    if not grp:
        continue
    A = np.array([r[2] for r in grp])
    B = np.array([r[3] for r in grp])
    dTA = np.array([r[4] for r in grp])
    dAB = np.array([r[5] for r in grp])
    print('%s n=%d  cubeA_x[%.3f,%.3f] cubeA_y[%.3f,%.3f]  cubeB_x[%.3f,%.3f] cubeB_y[%.3f,%.3f]  dTA mean=%.3f  dAB mean=%.3f'
          % (name, len(grp), A[:, 0].min(), A[:, 0].max(), A[:, 1].min(), A[:, 1].max(),
             B[:, 0].min(), B[:, 0].max(), B[:, 1].min(), B[:, 1].max(),
             dTA.mean(), dAB.mean()))
    for r in grp:
        print('   ep%2d cubeA=(%.3f,%.3f) cubeB=(%.3f,%.3f) dTA=%.3f dAB=%.3f' %
              (r[0], r[2][0], r[2][1], r[3][0], r[3][1], r[4], r[5]))
