import ipdb
from dm_control import suite
from dm_env import specs
import numpy as np
from gym import spaces
import gym
import random

def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)

def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

class CheetahEnv(gym.Env):
    def __init__(self):
        super(CheetahEnv, self).__init__()
        self.env = self.env = suite.load(domain_name="cheetah", task_name='run')
        self._max_episode_steps = 100
        self.action_space = _spec_to_box([self.env.action_spec()], np.float32)
        self._true_action_space = _spec_to_box([self.env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        self.observation_space = _spec_to_box(
            self.env.observation_spec().values(),
            np.float64
        )
        self.step_count = 0
        # reset the task
        self._task = self.reset_task()


    def step(self, action):
        """
        Execute one step in the environment.
        Should return: state, reward, done, info
        where info has to include a field 'task'.
        """


        assert self._true_action_space.contains(action)
        # reward = 0
        info = {'task': self._task.copy()}
        time_step = self.env.step(action)
        done = time_step.last()
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True
        # reward += time_step.reward or 0

        obs = self._get_obs(time_step)
        return obs, time_step.reward, done, info

    def reset(self):
        """
        Reset the environment. This should *NOT* automatically reset the task!
        Resetting the task is handled in the varibad wrapper (see wrappers.py).
        """
        self.step_count = 0
        time_step = self.env.reset()
        obs = self._get_obs(time_step)
        return obs

    def _get_obs(self, time_step):
        obs = _flatten_obs(time_step.observation)
        return obs

    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        return self._task.copy()

    def reset_task(self, task=None):
        if task is None:
            self._task = np.random.uniform(1, 20, 1) ## replace with with sample from range
        else:
            self._task = np.array(task)

        return self._task

# exam = CheetahEnv()
# print(exam._true_action_space)
