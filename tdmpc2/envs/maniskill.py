import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ManiSkill2 (0.4.x) registers its environments on the legacy gym (0.21)
# registry, so the env is created through that registry and then exposed
# through the gymnasium API used by the rest of TD-MPC2.
import gym as legacy_gym

from envs.wrappers.timeout import Timeout

import mani_skill2.envs


MANISKILL_TASKS = {
	'lift-cube': dict(
		env='LiftCube-v0',
		control_mode='pd_ee_delta_pos',
	),
	'pick-cube': dict(
		env='PickCube-v0',
		control_mode='pd_ee_delta_pos',
	),
	'stack-cube': dict(
		env='StackCube-v0',
		control_mode='pd_ee_delta_pos',
	),
	'pick-ycb': dict(
		env='PickSingleYCB-v0',
		control_mode='pd_ee_delta_pose',
	),
	'turn-faucet': dict(
		env='TurnFaucet-v0',
		control_mode='pd_ee_delta_pose',
	),
}


def as_gymnasium_space(space):
	"""Convert a legacy gym space into the equivalent gymnasium space."""
	if isinstance(space, spaces.Space):
		return space
	if hasattr(space, 'spaces'):
		return spaces.Dict({k: as_gymnasium_space(v) for k, v in space.spaces.items()})
	dtype = space.dtype
	low = np.asarray(space.low, dtype=dtype)
	high = np.asarray(space.high, dtype=dtype)
	try:
		return spaces.Box(low=low, high=high, dtype=dtype)
	except ValueError:
		return spaces.Box(
			low=np.broadcast_to(low, space.shape).copy(),
			high=np.broadcast_to(high, space.shape).copy(),
			dtype=dtype,
		)


class ManiSkillWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.observation_space = as_gymnasium_space(self.env.observation_space)
		self.action_space = spaces.Box(
			low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
			high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
			dtype=self.env.action_space.dtype,
		)

	def reset(self):
		return self.env.reset()

	def step(self, action):
		reward = 0
		for _ in range(2):
			obs, r, done, info = self.env.step(action)
			reward += r
			# ManiSkill2 tasks have no termination conditions (only time limits),
			# matching the protocol used in the TD-MPC2 paper.
			info['terminated'] = False
			# ManiSkill2 0.4.1 terminates the episode as soon as the task succeeds
			# (SapienEnv.get_done returns bool(info['success'])). The TD-MPC2
			# protocol (episodic=false) uses fixed-length episodes without early
			# termination, so success must not end the episode; only the Timeout
			# wrapper above is allowed to end it.
			done = False
			if done:
				break
		return obs, reward, done, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, args=None, **kwargs):
		frames = self.env.render(mode='cameras')
		if isinstance(frames, dict):
			frames = next(iter(frames.values()))
		return frames


def make_env(cfg):
	"""
	Make ManiSkill2 environment.
	"""
	if cfg.task not in MANISKILL_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'
	task_cfg = MANISKILL_TASKS[cfg.task]
	env = legacy_gym.make(
		task_cfg['env'],
		obs_mode='state',
		control_mode=task_cfg['control_mode'],
		render_camera_cfgs=dict(width=384, height=384),
	)
	env = ManiSkillWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=100)
	return env
