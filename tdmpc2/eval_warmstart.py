# -*- coding: utf-8 -*-
"""Evaluate warm-start checkpoint: policy-only and policy+MPPI."""
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

CKPT = _HERE.parent / "checkpoints" / "warmstart.pt"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
USE_MPPI = (len(sys.argv) > 2 and sys.argv[2] == "mppi")
CFG_SAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 256
CFG_HORIZON = int(sys.argv[4]) if len(sys.argv) > 4 else 3


def base_cfg():
    cfg = OmegaConf.load(_HERE / "config.yaml")
    if "defaults" in cfg:
        del cfg["defaults"]
    cfg.task = "stack-cube"; cfg.obs = "state"; cfg.episodic = False
    cfg.multitask = False; cfg.steps = 1; cfg.seed = 1; cfg.model_size = 1
    cfg.exp_name = "evalws"; cfg.compile = False; cfg.enable_wandb = False
    cfg.wandb_project = None; cfg.wandb_entity = None; cfg.save_video = False
    cfg.save_agent = False; cfg.save_csv = False
    cfg.mpc = USE_MPPI; cfg.iterations = 6; cfg.num_samples = CFG_SAMPLES
    cfg.num_elites = 32; cfg.num_pi_trajs = 16; cfg.horizon = CFG_HORIZON
    cfg.eval_freq = 10000; cfg.eval_episodes = 1
    cfg.buffer_size = 100000; cfg.batch_size = 256
    return cfg


set_seed(1)
cfg = base_cfg()
env = make_env(cfg)
cfg = parse_cfg(cfg)
agent = TDMPC2(cfg)
agent.load(CKPT)

wins, rewards = 0, []
for ep in range(N):
    obs = env.reset(); done = False; t = 0; er = 0.0
    while not done:
        a = agent.act(obs, t0=t == 0, eval_mode=True)
        obs, r, done, info = env.step(a)
        er += float(r); t += 1
    if info["success"] > 0.5:
        wins += 1
    rewards.append(er)
print(f"{'MPPI' if USE_MPPI else 'policy-only'}  N={N}  "
      f"SR={wins/N:.0%}  mean_ep_reward={np.mean(rewards):.1f}")
