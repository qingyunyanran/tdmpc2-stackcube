# -*- coding: utf-8 -*-
"""
Standalone CPU-friendly TD-MPC2 training entry for ManiSkill3 StackCube-v1.

Run from this directory (same layout as the official train.py):
    python train_stackcube.py --steps 1000000
"""
import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("LAZY_LEGACY_OP", "0")
warnings.filterwarnings("ignore")

import torch
from omegaconf import OmegaConf

_HERE = Path(__file__).resolve().parent

# parse_cfg calls hydra.utils.get_original_cwd(); patch it before import
import hydra.utils  # noqa: E402
hydra.utils.get_original_cwd = lambda: str(_HERE)

from common.parser import parse_cfg  # noqa: E402
from common.seed import set_seed  # noqa: E402
from common.buffer import Buffer  # noqa: E402
from envs import make_env  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402
from common.logger import Logger  # noqa: E402
from trainer.online_trainer import OnlineTrainer  # noqa: E402

torch.set_float32_matmul_precision("high")


def build_cfg(args):
    cfg = OmegaConf.load(_HERE / "config.yaml")
    if "defaults" in cfg:
        del cfg["defaults"]

    cfg.task = "stack-cube"
    cfg.obs = "state"
    cfg.episodic = False
    cfg.multitask = False
    cfg.steps = args.steps
    cfg.seed = args.seed
    cfg.model_size = args.model_size
    cfg.exp_name = args.exp_name

    # CPU: torch.compile on Windows CPU is slow/unstable -> disabled
    cfg.compile = False
    cfg.enable_wandb = False
    cfg.wandb_project = None
    cfg.wandb_entity = None
    cfg.save_video = False
    cfg.save_agent = True
    cfg.save_csv = True

    cfg.mpc = not args.no_mpc
    cfg.iterations = args.iterations
    cfg.num_samples = args.num_samples
    cfg.num_elites = args.num_elites
    cfg.num_pi_trajs = args.num_pi_trajs
    cfg.horizon = args.horizon

    cfg.eval_freq = args.eval_freq
    cfg.eval_episodes = args.eval_episodes
    cfg.buffer_size = args.buffer_size
    cfg.batch_size = args.batch_size
    return cfg


class SavingTrainer(OnlineTrainer):
    """OnlineTrainer with local checkpoints + periodic resume snapshots."""

    def __init__(self, *args, ckpt_dir=None, save_frequency=10_000,
                 warm_start=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._ckpt_dir = Path(ckpt_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._save_frequency = save_frequency
        self._best_sr = -1.0
        self._warm_start = warm_start

    def eval(self):
        metrics = super().eval()
        sr = metrics["episode_success"]
        if sr >= self._best_sr:
            self._best_sr = sr
            self.agent.save(self._ckpt_dir / "best.pt")
        return metrics

    def train(self):
        """OnlineTrainer.train with warm-start support + resume snapshots."""
        train_metrics, done, eval_next = {}, True, False
        # With warm-start the model is already trained: collect a small buffer
        # with the learned policy and start online updates immediately.
        skip_seed = self._warm_start
        while self._step <= self.cfg.steps:
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, 'eval')
                    eval_next = False

                if self._step > 0:
                    # ManiSkill3 sets terminated = success; this is a normal
                    # successful episode end, not an error.
                    train_metrics.update(
                        episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
                        episode_success=info['success'],
                        episode_length=len(self._tds),
                        episode_terminated=info['terminated'])
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))
                    if self._step % self._save_frequency < self.cfg.episode_length:
                        self.agent.save(self._ckpt_dir / "resume.pt")

                obs = self.env.reset()
                self._tds = [self.to_td(obs)]

            if skip_seed or self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)
            self._tds.append(self.to_td(obs, action, reward, info['terminated']))

            if skip_seed:
                # once the first episode is stored in the buffer, update online
                if self._ep_idx >= 1:
                    _train_metrics = self.agent.update(self.buffer)
                    train_metrics.update(_train_metrics)
            elif self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print('Pretraining agent on seed data...')
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += 1

        self.logger.finish(self.agent)
        self.agent.save(self._ckpt_dir / "final.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model-size", type=int, default=1, choices=[1, 5, 19])
    p.add_argument("--exp-name", type=str, default="cpu_run")
    p.add_argument("--no-mpc", action="store_true")
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--num-elites", type=int, default=32)
    p.add_argument("--num-pi-trajs", type=int, default=16)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--save-frequency", dest="save_frequency", type=int, default=10_000)
    p.add_argument("--eval-frequency", dest="eval_freq", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warm-start", type=str, default=None,
                   help="path to warm-start checkpoint (e.g. checkpoints/warmstart.pt)")
    args = p.parse_args()

    cfg = build_cfg(args)
    set_seed(cfg.seed)

    env = make_env(cfg)          # sets obs_shape/action_dim/episode_length on cfg
    cfg = parse_cfg(cfg)         # model size + derived fields
    print("Work dir:", cfg.work_dir)

    agent = TDMPC2(cfg)
    if args.warm_start:
        ws = Path(args.warm_start)
        if not ws.is_absolute():
            ws = _HERE.parent / ws
        agent.load(ws)
        print("Loaded warm-start checkpoint:", ws)
    buffer = Buffer(cfg)
    logger = Logger(cfg)
    ckpt_dir = _HERE.parent / "checkpoints" / f"seed{args.seed}_{args.exp_name}"
    trainer = SavingTrainer(
        cfg=cfg, env=env, agent=agent, buffer=buffer, logger=logger,
        ckpt_dir=ckpt_dir, save_frequency=args.save_frequency,
        warm_start=bool(args.warm_start),
    )
    trainer.train()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    main()
