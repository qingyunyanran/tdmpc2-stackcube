"""ManiSkill3 (3.0.x) environment adapter for TD-MPC2.

This mirrors ``envs/maniskill.py`` (ManiSkill2 0.4.x) as closely as possible so
that the algorithm, hyper-parameters and episode protocol are unchanged:

  * ``obs_mode='state'``          -> flat state vector (48-dim for StackCube-v1)
  * ``control_mode='pd_ee_delta_pos'`` -> 4-dim action in [-1, 1] (delta pos +-0.1 m)
  * ``reward_mode='dense'``       -> same un-normalised dense reward as MS2
  * action repeat of 2 env steps per agent step (paper protocol)
  * fixed-length episodes: neither task success nor the internal time limit may
    terminate the episode, only the outer ``Timeout`` wrapper does.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

from envs.wrappers.timeout import Timeout


MANISKILL3_TASKS = {
	'stack-cube': dict(
		env='StackCube-v1',
		control_mode='pd_ee_delta_pos',
	),
	'pick-cube': dict(
		env='PickCube-v1',
		control_mode='pd_ee_delta_pos',
	),
	'lift-cube': dict(
		env='LiftCube-v1',
		control_mode='pd_ee_delta_pos',
	),
	'push-cube': dict(
		env='PushCube-v1',
		control_mode='pd_ee_delta_pos',
	),
}

# Paper protocol: episode length 200 env steps == 100 agent steps @ repeat 2.
ENV_EPISODE_STEPS = 200
ACTION_REPEAT = 2


class ManiSkill3Wrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.cfg = cfg
		# Uniform [-1, 1] action box, same as the ManiSkill2 adapter.
		self.action_space = spaces.Box(
			low=np.full(self.env.action_space.shape, -1.0, dtype=np.float32),
			high=np.full(self.env.action_space.shape, 1.0, dtype=np.float32),
			dtype=np.float32,
		)

	def reset(self, **kwargs):
		# ManiSkill3 follows the gymnasium 5-tuple API; TD-MPC2 expects obs only.
		obs, _ = self.env.reset(**kwargs)
		return obs

	def step(self, action):
		reward = 0.0
		for _ in range(ACTION_REPEAT):
			obs, r, terminated, truncated, info = self.env.step(action)
			reward += float(r)
			# ManiSkill3's BaseEnv.step sets terminated = info['success'], which
			# would end the episode as soon as the task succeeds. The TD-MPC2
			# protocol (episodic=false) uses fixed-length episodes without early
			# termination, so both success-termination and ManiSkill3's internal
			# time limit are suppressed here; episode length is enforced solely
			# by the Timeout wrapper below (100 agent steps == 200 env steps).
			info['terminated'] = False
		return obs, reward, False, info

	def render(self, args=None, **kwargs):
		return self.env.render()


def make_env(cfg):
	"""Make ManiSkill3 environment."""
	if cfg.task not in MANISKILL3_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'
	task_cfg = MANISKILL3_TASKS[cfg.task]
	env = gym.make(
		task_cfg['env'],
		num_envs=1,
		obs_mode='state',
		control_mode=task_cfg['control_mode'],
		reward_mode='dense',
		robot_uids='panda',
		sim_backend='cpu',
		render_mode='rgb_array',
		max_episode_steps=ENV_EPISODE_STEPS,
	)
	env = CPUGymWrapper(env)
	env = ManiSkill3Wrapper(env, cfg)
	env = Timeout(env, max_episode_steps=ENV_EPISODE_STEPS // ACTION_REPEAT)
	return env

