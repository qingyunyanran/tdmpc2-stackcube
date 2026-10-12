import os
import sys

import gymnasium as gym
import numpy as np
from envs.wrappers.timeout import Timeout

import mani_skill.envs  # noqa: F401  (registers ManiSkill3 tasks)

# Reuse the 42-dim unified-obs builder from the CL project
_CL_DIR = r"E:\munichi\maniskill-cl\cl"
if _CL_DIR not in sys.path:
    sys.path.insert(0, _CL_DIR)
from unified_obs import build_unified_obs, SLOT_LAYOUT  # noqa: E402

MANISKILL_TASKS = {
    'stack-cube': dict(
        env='StackCube-v1',
        control_mode='pd_ee_delta_pos',
    ),
}


class ManiSkill3Wrapper(gym.Wrapper):
    """ManiSkill3 (num_envs=1, state_dict obs) -> single-agent classic-style env.

    - obs: 42-dim unified float32 vector
    - action: symmetric Box matching TD-MPC2's [-1,1] action assumption
    - step returns the (obs, r, done, info) 4-tuple the trainer expects
    """

    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self._slot_cfg = SLOT_LAYOUT['StackCube-v1']
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32)
        act_low = float(self.env.action_space.low.min())
        act_high = float(self.env.action_space.high.max())
        self.action_space = gym.spaces.Box(
            low=np.full((4,), act_low, dtype=np.float32),
            high=np.full((4,), act_high, dtype=np.float32),
        )

    def _unified(self, obs):
        return build_unified_obs(obs, self._slot_cfg)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._unified(obs)

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)[None]  # (1, 4)
        obs, reward, terminated, truncated, info = self.env.step(a)
        uobs = self._unified(obs)
        r = float(reward.reshape(-1)[0])
        done = bool(terminated.reshape(-1)[0]) or bool(truncated.reshape(-1)[0])
        out_info = {
            'success': float(info['success'].reshape(-1)[0]),
            'terminated': float(terminated.reshape(-1)[0]),
        }
        return uobs, r, done, out_info

    @property
    def unwrapped(self):
        return self.env.unwrapped


def make_env(cfg):
    """Make ManiSkill3 StackCube-v1 environment."""
    if cfg.task not in MANISKILL_TASKS:
        raise ValueError('Unknown task:', cfg.task)
    assert cfg.obs == 'state', 'This task only supports state observations.'
    task_cfg = MANISKILL_TASKS[cfg.task]
    env = gym.make(
        task_cfg['env'],
        num_envs=1,
        obs_mode='state_dict',
        control_mode=task_cfg['control_mode'],
        sim_backend='cpu',
        reward_mode='dense',
        reconfiguration_freq=1,
    )
    env = ManiSkill3Wrapper(env, cfg)
    env = Timeout(env, max_episode_steps=50)
    return env
