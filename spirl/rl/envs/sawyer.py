import numpy as np
from collections import defaultdict
import d4rl
from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from mujoco_py.builder import MujocoException
import gym
from gym import wrappers
import torch
from torchvision.transforms import Resize
from spirl.utils.pytorch_utils import ten2ar
from contextlib import contextmanager
import os.path as osp
from gym import spaces
from gym.spaces import Box
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from PIL import Image

SUPPORTED_SAWYER_ENVS = ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2', 'door-close-v2',
                         'drawer-open-v2', 'drawer-close-v2', 'button-press-topdown-v2', 'button-press-v2',
                         'peg-insert-side-v2', 'window-open-v2', 'window-close-v2', 'sweep-v2',
                         'basketball-v2', 'shelf-place-v2', 'sweep-into-v2', 'lever-pull-v2']


# TODO: check done on timeout -> should not give an internal 
# TODO: check that completed subtasks has no effect on training cycle -> should be okay
class SawyerEnv:
    """Tiny wrapper around GymEnv for Kitchen tasks."""
    SUBTASKS = []

    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(self._hp.name)
        self._mj_except = MujocoException
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[self._hp.name + '-goal-observable']
        env = env_cls(seed=0)
        fix_seed = False
        if fix_seed:
            env.random_init = False
        # Some magic that is performed in the example script from metaworld
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.reset()
        env.reset_model()  # Might not be necessary
        self._env = env
        self._env_name = self._hp.name
        self._short_name = 'sawyer'
        self.terminate_on_success = False
        self.goal_cond = True
        self.visual = False
        self.neg_rew = False
        # self.max_steps = 500
        self.obs = None

        if self.visual:
            self.observation_space = spaces.Box(0., 1., [3, 64, 64], dtype='float32')
            self.goal_cond=False
        else:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32)

    def _get_obs(self):
        """ Returns the current observation. """
        if self.visual:
            obs = self._env.sim.render(64, 64, mode='offscreen', camera_name='corner')[:,:,::-1].astype(np.float32) / 255.
            return np.moveaxis(obs, -1, 0)
        return self.obs

    def __getattr__(self, attr):
        """ Handles wrapping. """
        return getattr(self._env, attr)

    def compute_reward(self, achieved_goal, goal, info):
        """
        Computes rewards according to currently achieved and desired goal.
        metaworld did not expose this function, so it had to be extracted manually.
        Args:
            achieved_goal : 1D or 2D array containing one or a batch of achieved goals.
            goal : 1D or 2D array containing one or a batch of desired goals.
            info (dict): Additional information (currently not in use).
        """
        if self._env_name == 'reach-v2':
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.06).astype(np.float32)
            # reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.05).astype(np.float32)
        elif self._env_name == 'door-open-v2':
            reward = (np.transpose(np.abs(achieved_goal - goal))[0] <= 0.08).astype(np.float32)
        elif self._env_name == 'door-close-v2':
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.08).astype(np.float32)
        elif self._env_name in ['window-open-v2', 'window-close-v2']:
            reward = (np.transpose(np.abs(achieved_goal - goal))[0] <= 0.05).astype(np.float32)
        elif self._env_name in ['push-v2', 'sweep-v2']:
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.05).astype(np.float32)
        elif self._env_name in ['pick-place-v2', 'shelf-place-v2']:
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.07).astype(np.float32)
        elif self._env_name == 'drawer-open-v2':
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.03).astype(np.float32)
        elif self._env_name == 'drawer-close-v2':
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.055).astype(np.float32)
        elif self._env_name == 'button-press-topdown-v2':
            reward = (np.transpose(np.abs(achieved_goal - goal))[2] <= 0.02).astype(np.float32)
        elif self._env_name == 'button-press-v2':
            reward = (np.transpose(np.abs(achieved_goal - goal))[1] <= 0.02).astype(np.float32)
        elif self._env_name == 'basketball-v2':
            goal = goal.copy()
            goal[2] = 0.3
            target_to_obj = (achieved_goal - goal) * np.array([[1., 1., 2.]])
            reward = (np.linalg.norm(target_to_obj, axis=-1)  <= 0.08).astype(np.float32)
        elif self._env_name == 'sweep-into-v2':
            goal = goal.copy()
            goal[:, 2] = achieved_goal[:, 2]
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.05).astype(np.float32)
        elif self._env_name == 'lever-pull-v2':
            raise Exception ('Sadly this reward is not easily computable from states.')
        elif self._env_name == 'peg-insert-side-v2':
            raise Exception ('Reward computation requires access to simlation.')
        else:
            raise Exception(f'{self._env_name} not implemented yet.')
        return reward

    def reset(self):
        self.solved_subtasks = defaultdict(lambda: 0)
        self.obs = self._env.reset()
        return self._wrap_observation(self._get_obs())

    def step(self, action):
        if isinstance(action, torch.Tensor): action = ten2ar(action)
        try:
            self.obs, _, done, info = self._env.step(action)
            if self.goal_cond and self._env_name != 'peg-insert-side-v2':
                achieved_goal = self.obs[:3].copy() if self._env_name == 'reach-v2' else self.obs[4:7].copy()
                desired_goal = self.obs[36:].copy()
                reward = self.compute_reward(achieved_goal, desired_goal, {})
            else:
                reward = 1.0 if info['success'] else 0.0  # Reward signal is adapted to be sparse.
            done = False #True if ((self._env.curr_path_length >= self.max_steps) or (self.terminate_on_success and reward > 0)) else done
            reward = reward - 1 if self.neg_rew else reward
            reward = reward / self._hp.reward_norm
        except self._mj_except:
            # this can happen when agent drives simulation to unstable region (e.g. very fast speeds)
            print("Catch env exception!")
            obs = self.reset()
            reward = self._hp.punish_reward     # this avoids that the agent is going to these states again
            done = np.array(True)        # terminate episode (observation will get overwritten by env reset)
            info = {}
        return self._wrap_observation(self._get_obs()), np.float64(reward), np.array(done), self._postprocess_info(info)

    def render(self, mode='rgb_array'):
        # TODO make env render in the correct size instead of downsizing after for performance
        img = Resize((self._hp.screen_height, self._hp.screen_width))(Image.fromarray(self._render_raw(mode=mode)))
        return np.array(img) / 255.

    def _make_env(self, id):
        """Instantiates the environment given the ID."""
        env = gym.make(id)
        if isinstance(env, wrappers.TimeLimit) and self._hp.unwrap_time:
            # unwraps env to avoid this bug: https://github.com/openai/gym/issues/1230
            env = env.env
        return env

    def get_episode_info(self):
        """Allows to return logging info about latest episode (sindce last reset)."""
        info = AttrDict()
        if hasattr(self._env, "get_episode_info"):
            info = self._env.get_episode_info()
        # info.update(AttrDict(self.solved_subtasks))
        return info

    def _render_raw(self, mode):
        """Returns rendering as uint8 in range [0...255]"""
        return self._env.render(mode=mode)

    def _default_hparams(self):
        default_dict = ParamDict({
            'device': None,         # device that all tensors should get transferred to
            'screen_width': 64,     # width of rendered images
            'screen_height': 64,    # height of rendered images
            'name': self._name,   # name of openai/gym environment
            'reward_norm': 1.,  # reward normalization factor
            'punish_reward': -100,   # reward used when action leads to simulation crash
            'unwrap_time': True,    # removes time limit wrapper from envs so that done is not set on timeout
        })
        return default_dict

    def _postprocess_info(self, info):
        """Sorts solved subtasks into separately logged elements."""
        # completed_subtasks = info.pop("completed_tasks")
        # for task in self.SUBTASKS:
        #     self.solved_subtasks[task] = 1 if task in completed_subtasks or self.solved_subtasks[task] else 0
        return info

    @contextmanager
    def val_mode(self):
        """Sets validation parameters if desired. To be used like: with env.val_mode(): ...<do something>..."""
        pass; yield; pass

    def _wrap_observation(self, obs):
        """Process raw observation from the environment before return."""
        return np.asarray(obs, dtype=np.float32)

    @property
    def agent_params(self):
        """Parameters for agent that can be handed over after env is constructed."""
        return AttrDict()
