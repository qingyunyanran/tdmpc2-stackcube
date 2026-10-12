# -*- coding: utf-8 -*-
"""Report per-dim post-tanh policy action MSE on demos for a checkpoint."""
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

CKPT = sys.argv[1]
if not Path(CKPT).is_absolute():
    CKPT = str(_HERE.parent / CKPT)

# reuse demo loader
sys.path.insert(0, str(_HERE))
from warmstart import load_demos, DEMO_PATH

cfg0 = OmegaConf.load(_HERE / "config.yaml")
if "defaults" in cfg0:
    del cfg0["defaults"]
cfg0.task = "stack-cube"; cfg0.obs = "state"; cfg0.episodic = False
cfg0.multitask = False; cfg0.steps = 1; cfg0.seed = 1; cfg0.model_size = 1
cfg0.exp_name = "x"; cfg0.compile = False; cfg0.enable_wandb = False
cfg0.wandb_project = None; cfg0.wandv_entity = None; cfg0.save_video = False
cfg0.save_agent = False; cfg0.save_csv = False
cfg0.mpc = False; cfg0.iterations = 6; cfg0.num_samples = 256
cfg0.num_elites = 32; cfg0.num_pi_trajs = 16; cfg0.horizon = 3
cfg0.eval_freq = 10000; cfg0.eval_episodes = 1
cfg0.buffer_size = 100000; cfg0.batch_size = 256

set_seed(1)
env = make_env(cfg0)
cfg = parse_cfg(cfg0)
agent = TDMPC2(cfg)
agent.load(CKPT)
model = agent.model; dev = agent.device

O, A, OP = load_demos(DEMO_PATH)
with torch.no_grad():
    o = torch.from_numpy(O).to(dev)
    z = model.encode(o, None)
    mean_pre = model._pi(z).chunk(2, dim=-1)[0]
    pred = torch.tanh(mean_pre).cpu().numpy()

err = (pred - A) ** 2
print(f"checkpoint: {CKPT}")
print(f"overall post-tanh MSE = {err.mean():.5f}  RMSE = {err.mean()**0.5:.4f}")
names = ["dx", "dy", "dz", "gripper"]
for i in range(4):
    print(f"  dim {names[i]:8s}: MSE={err[:,i].mean():.5f}  MAE={np.abs(pred[:,i]-A[:,i]).mean():.4f}")
