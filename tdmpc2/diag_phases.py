# -*- coding: utf-8 -*-
"""Phase-level failure diagnosis for the current best TD-MPC2 checkpoint."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import hydra.utils
_HERE = Path(__file__).resolve().parent
hydra.utils.get_original_cwd = lambda: str(_HERE)

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

CKPT = _HERE.parent / "checkpoints" / "seed1_cpu300k" / "best.pt"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
_ns = int(sys.argv[2]) if len(sys.argv) > 2 else 256
_hz = int(sys.argv[3]) if len(sys.argv) > 3 else 3

cfg = OmegaConf.load(_HERE / "config.yaml")
if "defaults" in cfg:
    del cfg["defaults"]
cfg.task = "stack-cube"; cfg.obs = "state"; cfg.episodic = False
cfg.multitask = False; cfg.steps = 1; cfg.seed = 1; cfg.model_size = 1
cfg.exp_name = "diag"; cfg.compile = False; cfg.enable_wandb = False
cfg.wandb_project = None; cfg.wandb_entity = None; cfg.save_video = False
cfg.save_agent = False; cfg.save_csv = False
cfg.mpc = True; cfg.iterations = 6; cfg.num_samples = _ns
cfg.num_elites = max(16, _ns // 8); cfg.num_pi_trajs = 16; cfg.horizon = _hz
cfg.eval_freq = 10000; cfg.eval_episodes = 1
cfg.buffer_size = 100000; cfg.batch_size = 256

set_seed(1)
env = make_env(cfg)
cfg = parse_cfg(cfg)
agent = TDMPC2(cfg)
agent.load(CKPT)
print("loaded", CKPT, flush=True)

# unified obs: cubeA xyz = [25:28], cubeB xyz = [32:35]
stats = dict(grasped=0, lifted5cm=0, above=0, aligned=0, success=0)
max_h, min_dx, min_dz = [], [], []
grasp_step_dist = []

for ep in range(N):
    obs = env.reset()
    done = False; t = 0
    ep_lifted = False; ep_above = False; ep_aligned = False; ep_grasped = False
    ep_maxh = 0.0; ep_min_dx = 9.0; ep_min_dz = 9.0
    ep_min_tcp = 9.0
    while not done:
        a = agent.act(obs, t0=t == 0, eval_mode=True)
        # tcp-cubeA distance (tcp [18:21])
        d_tcp = float(np.linalg.norm(np.asarray(obs[18:21]) - np.asarray(obs[25:28])))
        ep_min_tcp = min(ep_min_tcp, d_tcp)
        obs, r, done, info = env.step(a)
        obs_np = np.asarray(obs)
        az = float(obs_np[27]); bx, by, bz = obs_np[32:35]
        ax, ay = obs_np[25:27]
        ep_maxh = max(ep_maxh, az)
        if az > 0.06:
            ep_lifted = True
        if az > 0.06 and a[3] < 0.0:  # gripper closed heuristic not available post-step; skip
            pass
        # horizontal/vertical gap (cube half-size ~0.02; b top ~ bz+0.02)
        dx = float(np.linalg.norm([ax - bx, ay - by]))
        dz = az - bz
        if az > 0.08:  # only meaningful once lifted
            ep_min_dx = min(ep_min_dx, dx)
            ep_min_dz = min(ep_min_dz, dz)
            if dx < 0.04 and dz < 0.08:
                ep_aligned = True
            if dz < 0.08 and dz > 0.0 and dx < 0.05:
                ep_above = True
        t += 1
    if ep_min_tcp < 0.05:
        ep_grasped = True
    if info["success"] > 0.5:
        stats["success"] += 1
    stats["grasped"] += int(ep_grasped)
    stats["lifted5cm"] += int(ep_lifted)
    stats["above"] += int(ep_above)
    stats["aligned"] += int(ep_aligned)
    max_h.append(ep_maxh); min_dx.append(ep_min_dx); min_dz.append(ep_min_dz)
    grasp_step_dist.append(ep_min_tcp)

print(f"\n=== {N} episodes ===")
for k, v in stats.items():
    print(f"{k:10s}: {v}/{N} = {v/N:.0%}")
print(f"cubeA max height      : mean {np.mean(max_h):.3f}  max {np.max(max_h):.3f}")
print(f"min horiz gap (lifted): mean {np.nanmean([x for x in min_dx if x<9]):.3f}  best {np.min(min_dx):.3f}")
print(f"min vert gap (lifted) : mean {np.nanmean([x for x in min_dz if x<9]):.3f}  best {np.min(min_dz):.3f}")
print(f"min tcp-cubeA dist    : mean {np.mean(grasp_step_dist):.3f}  best {np.min(grasp_step_dist):.3f}")
