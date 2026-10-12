# -*- coding: utf-8 -*-
"""
Expert warm-start for TD-MPC2 on ManiSkill3 StackCube-v1.

Supervises two components using the 889 official successful demonstrations:
  (1) latent consistency:  dynamics(encode(o_t), a_t) ~ encode(o_{t+1})
  (2) policy BC:           pi_mean(encode(o_t)) ~ a_t   (pre-squash MSE)
After warm-start, the saved checkpoint can be loaded by train_stackcube.py
(via --warm-start) and refined with online TD-MPC2 updates.
"""
import argparse
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path

import numpy as np
import h5py
import torch

from omegaconf import OmegaConf
import hydra.utils
_HERE = Path(__file__).resolve().parent
hydra.utils.get_original_cwd = lambda: str(_HERE)

from common.parser import parse_cfg       # noqa: E402
from common.seed import set_seed        # noqa: E402
from envs import make_env               # noqa: E402
from tdmpc2 import TDMPC2               # noqa: E402

DEMO_PATH = r"C:\Users\Administrator\.maniskill\demos\StackCube-v1\rl\trajectory.state.pd_ee_delta_pos.physx_cpu.h5"


def flat_state_to_unified(raw):
    u = np.zeros(42, np.float32)
    u[0:18] = raw[0:18]
    u[18:25] = raw[18:25]
    u[25:32] = raw[25:32]
    u[32:39] = raw[32:39]
    return u


def load_demos(path, max_traj=None):
    """Return arrays of (obs_t, action, obs_tp1) pairs across all demos."""
    f = h5py.File(path, "r")
    keys = sorted(f.keys(), key=lambda s: int(s.split("_")[1]))
    if max_traj:
        keys = keys[:max_traj]
    ot, a, otp = [], [], []
    for k in keys:
        t = f[k]
        actions = np.clip(t["actions"][:], -1.0, 1.0).astype(np.float32)
        raw = t["obs"][:]
        T = len(actions)
        u = np.stack([flat_state_to_unified(raw[i]) for i in range(T + 1)])
        ot.append(u[:T]); a.append(actions); otp.append(u[1:T + 1])
    f.close()
    return (np.concatenate(ot, 0), np.concatenate(a, 0), np.concatenate(otp, 0))


def build_base_cfg():
    cfg = OmegaConf.load(_HERE / "config.yaml")
    if "defaults" in cfg:
        del cfg["defaults"]
    cfg.task = "stack-cube"; cfg.obs = "state"; cfg.episodic = False
    cfg.multitask = False; cfg.steps = 1; cfg.seed = 1; cfg.model_size = 1
    cfg.exp_name = "warmstart"; cfg.compile = False; cfg.enable_wandb = False
    cfg.wandb_project = None; cfg.wandb_entity = None; cfg.save_video = False
    cfg.save_agent = False; cfg.save_csv = False
    cfg.mpc = True; cfg.iterations = 6; cfg.num_samples = 256
    cfg.num_elites = 32; cfg.num_pi_trajs = 16; cfg.horizon = 3
    cfg.eval_freq = 10000; cfg.eval_episodes = 1
    cfg.buffer_size = 100000; cfg.batch_size = 256
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dyn-weight", type=float, default=1.0)
    p.add_argument("--pi-weight", type=float, default=1.0)
    p.add_argument("--max-traj", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    set_seed(1)
    cfg = build_base_cfg()
    env = make_env(cfg)
    cfg = parse_cfg(cfg)

    agent = TDMPC2(cfg)
    model = agent.model
    device = agent.device

    print("loading demos...", flush=True)
    O, A, OP = load_demos(DEMO_PATH, args.max_traj)
    print(f"  {len(O)} transitions", flush=True)
    O = torch.from_numpy(O); A = torch.from_numpy(A); OP = torch.from_numpy(OP)

    # train encoder, dynamics, policy together
    params = list(model._encoder.parameters()) + \
             list(model._dynamics.parameters()) + \
             list(model._pi.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    n = len(O)
    # expert action target for policy MSE: invert tanh is unnecessary;
    # supervise the raw pre-squash policy mean with atanh(expert) target.
    target_pi = torch.atanh(A.clamp(-0.999, 0.999)).to(device)

    model.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        d_tot, p_tot, nb = 0.0, 0.0, 0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            o = O[idx].to(device); a = A[idx].to(device); op = OP[idx].to(device)

            z = model.encode(o, None)
            z_pred = model.next(z, a, None)
            with torch.no_grad():
                z_target = model.encode(op, None)
            # latent consistency (both are SimNorm outputs in [0,~simnorm_dim])
            d_loss = ((z_pred - z_target) ** 2).mean()

            raw = model._pi(z).chunk(2, dim=-1)[0]   # pre-squash mean
            p_loss = ((raw - target_pi[idx]) ** 2).mean()

            loss = args.dyn_weight * d_loss + args.pi_weight * p_loss
            opt.zero_grad(); loss.backward(); opt.step()

            d_tot += float(d_loss) * len(idx)
            p_tot += float(p_loss) * len(idx)
            nb += len(idx)
        if epoch % 2 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  dyn={d_tot/nb:.5f}  pi_mse={p_tot/nb:.5f}",
                  flush=True)

    out = args.out or str(_HERE.parent / "checkpoints" / "warmstart.pt")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out)
    print("saved warm-start checkpoint ->", out, flush=True)


if __name__ == "__main__":
    main()
