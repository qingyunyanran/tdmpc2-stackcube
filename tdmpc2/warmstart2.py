# -*- coding: utf-8 -*-
"""
Staged expert warm-start for TD-MPC2 on ManiSkill3 StackCube-v1.

Stage 1: train encoder + dynamics jointly (latent consistency on demos).
Stage 2: freeze encoder + dynamics; train ONLY the policy head to low MSE
         (exact behavior cloning in the now-stable latent space).
Output: checkpoints/warmstart2.pt loadable via train_stackcube.py --warm-start.
"""
import argparse
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path

import numpy as np
import h5py
import torch

from omegaconf import OmegaConf
import hydra.utils
_HERE = Path(__file__).resolve().parent
hydra.utils.get_original_cwd = lambda: str(_HERE)

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

DEMO_PATH = r"C:\Users\Administrator\.maniskill\demos\StackCube-v1\rl\trajectory.state.pd_ee_delta_pos.physx_cpu.h5"


def flat_state_to_unified(raw):
    u = np.zeros(42, np.float32)
    u[0:18] = raw[0:18]
    u[18:25] = raw[18:25]
    u[25:32] = raw[25:32]
    u[32:39] = raw[32:39]
    return u


def load_demos(path, max_traj=None):
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
    cfg.exp_name = "warmstart2"; cfg.compile = False; cfg.enable_wandb = False
    cfg.wandb_project = None; cfg.wandb_entity = None; cfg.save_video = False
    cfg.save_agent = False; cfg.save_csv = False
    cfg.mpc = True; cfg.iterations = 6; cfg.num_samples = 256
    cfg.num_elites = 32; cfg.num_pi_trajs = 16; cfg.horizon = 3
    cfg.eval_freq = 10000; cfg.eval_episodes = 1
    cfg.buffer_size = 100000; cfg.batch_size = 256
    return cfg


def run_stage(opt, params, model, device, O, A, OP, epochs, bs,
              kind, target_pi=None):
    n = len(O)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            o = O[idx].to(device); a = A[idx].to(device); op = OP[idx].to(device)
            z = model.encode(o, None)
            if kind == "dyn":
                z_pred = model.next(z, a, None)
                with torch.no_grad():
                    z_t = model.encode(op, None)
                loss = ((z_pred - z_t) ** 2).mean()
            else:  # policy-only BC
                raw = model._pi(z).chunk(2, dim=-1)[0]
                loss = ((raw - target_pi[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx); nb += len(idx)
        if epoch % 5 == 0 or epoch == epochs - 1:
            tag = "dyn" if kind == "dyn" else "pi_mse"
            print(f"  [{tag}] epoch {epoch:3d}  {tot/nb:.6f}", flush=True)
    return tot / nb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dyn-epochs", type=int, default=40)
    p.add_argument("--pi-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--pi-lr", type=float, default=1e-3)
    p.add_argument("--max-traj", type=int, default=None)
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
    target_pi = torch.atanh(A.clamp(-0.999, 0.999)).to(device)

    model.train()
    # ---- Stage 1: encoder + dynamics ----
    print("Stage 1: encoder + dynamics", flush=True)
    p1 = list(model._encoder.parameters()) + list(model._dynamics.parameters())
    opt1 = torch.optim.Adam(p1, lr=args.lr)
    run_stage(opt1, p1, model, device, O, A, OP, args.dyn_epochs,
              args.batch_size, "dyn")

    # ---- Stage 2: freeze encoder/dynamics, train policy only ----
    print("Stage 2: policy-only BC (encoder/dynamics frozen)", flush=True)
    for par in model._encoder.parameters():
        par.requires_grad_(False)
    for par in model._dynamics.parameters():
        par.requires_grad_(False)
    opt2 = torch.optim.Adam(model._pi.parameters(), lr=args.pi_lr)
    run_stage(opt2, model._pi.parameters(), model, device, O, A, OP,
              args.pi_epochs, args.batch_size, "pi", target_pi=target_pi)

    out = str(_HERE.parent / "checkpoints" / "warmstart2.pt")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out)
    print("saved ->", out, flush=True)


if __name__ == "__main__":
    main()
