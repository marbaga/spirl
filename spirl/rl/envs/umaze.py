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
from PIL import Image


class UMazeEnv:

    def __init__(self, config):
        # name, max_steps=500, corridor_length=15, terminate_on_success=False, neg_rew=False, goal_cond=False, visual=False):
        self._hp = self._default_hparams().overwrite(config)
        self._mj_except = MujocoException

        self._env_name = self._hp.env_name
        self._short_name = 'maze'
        self.max_steps = 500
        self.dist_threshold = 1.2
        self.state_size = 2
        self.action_size = 2
        self.goal_cond = True
        self.visual=False
        self.action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.neg_rew = False

        self.state = None
        self.goal = None
        self.t = None
        self.terminate_on_success = False
        if name == 'umaze-v1':
            self.sim = make_u_maze(60.0)
        elif name == 'crossmaze-v1':
            self.sim = make_cross_maze(42.0)
        elif name == 'custommaze-v1':
            self.sim = make_custom_maze()
        elif name == 'roommaze-v1':
            self.sim = make_room_maze(14.0)
        elif name =='rebuttalmaze-v1':
            self.sim = make_rebuttal_maze(6.0)
        else:
            raise NotImplementedError
        self.metadata = {'video.frames_per_second': 30}

        if self.visual:
            self.observation_space = spaces.Box(0., 1., [3, 64, 64], dtype='float32')
            self.goal_cond = False
        else:
            low = np.array([min(x) - .5 for x in zip(*[v['loc'] for k, v in self.sim._segments.items()])])
            high = np.array([max(x) + .5 for x in zip(*[v['loc'] for k, v in self.sim._segments.items()])])
            self.scale = lambda x: (x - low) / (high-low)
            self.unscale = lambda x: x * (high-low) + low
            if self.goal_cond: 
                self.observation_space = spaces.Dict(dict(
                    desired_goal=spaces.Box(low=0., high=1., shape=(2,), dtype='float32'),
                    achieved_goal=spaces.Box(low=0., high=1., shape=(2,), dtype='float32'),
                    observation=spaces.Box(low=0., high=1., shape=(2,), dtype='float32'),
                ))
            else:
                self.observation_space = Box(low=0., high=1., shape=(2,), dtype=np.float32)

    def _get_obs(self):
        """ Returns the current observation. """
        if self.visual:
            if self.canvas is None:
                # self.canvas contains a rendering of the maze structure. It is recomputed at each reset.
                # self.to_coord is a lambda projecting environment coordinates to pixel coordinates.
                self.canvas, self.to_coord = self.sim.get_canvas()
            # Render the agent's and the goal's positions
            enlarge = lambda pos: [(pos[0] - 1 + (i // 3), pos[1] - 1 + (i % 3)) for i in range(9)]
            clip = lambda l: [np.clip(e, 0, 63) for e in l]
            to_idx = lambda l: (tuple([e[0] for e in l]), tuple([e[1] for e in l]))
            canvas = self.canvas.copy()
            canvas[to_idx(clip(enlarge(self.to_coord(self.state))))] = [1., 0., 0.]
            canvas[to_idx(clip(enlarge(self.to_coord(self.goal))))] = [0., 1., 0.]
            return np.moveaxis(np.flip(canvas, 0), -1, 0)
        if not self.goal_cond:
            return self.scale(self.state.copy())
        return {
            'observation': self.scale(self.state.copy()),
            'achieved_goal': self.scale(self.state.copy()),
            'desired_goal': self.scale(self.goal.copy())
        }

    def compute_reward(self, achieved_goal, goal, info):
        """
        Computes rewards according to currently achieved and desired goal by measuring euclidean distances.
        Args:
            achieved_goal : 1D or 2D array containing one or a batch of achieved goals.
            goal : 1D or 2D array containing one or a batch of desired goals.
            info (dict): Additional information (currently not in use).
        """
        achieved_goal, goal = self.unscale(achieved_goal), self.unscale(goal)
        return (np.linalg.norm(achieved_goal - goal, axis=-1) <= self.dist_threshold).astype(np.float32)

    def reset(self):
        """ Resets the environment. """
        self.solved_subtasks = defaultdict(lambda: 0)
        self.state = np.array(self.sim.sample_start())
        self.goal = np.array(self.sim.sample_goal())
        self.t = 0
        self.canvas = None
        return self._wrap_observation(self._get_obs())

    def step(self, action):
        """ Advances simulation by a step. """
        if isinstance(action, torch.Tensor): action = ten2ar(action)
        try:
            self.t += 1
            self.state = self.sim.move((self.state[0], self.state[1]), (action[0], action[1]))
            self.state = np.array(self.state)
            if self.goal_cond:
                reward = self.compute_reward(self._get_obs()['achieved_goal'], self._get_obs()['desired_goal'], {})
            else:
                reward = (np.linalg.norm(self.state - self.goal, axis=-1) <= self.dist_threshold).astype(np.float32)
            done = (self.t >= self.max_steps) or (self.terminate_on_success and reward > 0)
            reward = reward - 1 if self.neg_rew else reward
            reward = reward / self._hp.reward_norm
        except self._mj_except:
            # this can happen when agent drives simulation to unstable region (e.g. very fast speeds)
            print("Catch env exception!")
            obs = self.reset()
            reward = self._hp.punish_reward     # this avoids that the agent is going to these states again
            done = np.array(True)        # terminate episode (observation will get overwritten by env reset)
            info = {}
        return self._wrap_observation(self._get_obs()), np.float64(reward), np.array(done), self._postprocess_info({})

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
        if self.canvas is None:
            # self.canvas contains a rendering of the maze structure. It is recomputed at each reset.
            # self.to_coord is a lambda projecting environment coordinates to pixel coordinates.
            self.canvas, self.to_coord = self.sim.get_canvas()
        # Render the agent's and the goal's positions
        enlarge = lambda pos: [(pos[0] - 1 + (i // 3), pos[1] - 1 + (i % 3)) for i in range(9)]
        clip = lambda l: [np.clip(e, 0, 63) for e in l]
        to_idx = lambda l: (tuple([e[0] for e in l]), tuple([e[1] for e in l]))
        canvas = self.canvas.copy()
        canvas[to_idx(clip(enlarge(self.to_coord(self.state))))] = [1., 0., 0.]
        canvas[to_idx(clip(enlarge(self.to_coord(self.goal))))] = [0., 1., 0.]
        return np.moveaxis(np.flip(canvas, 0), -1, 0)

    def _default_hparams(self):
        default_dict = ParamDict({
            'device': None,         # device that all tensors should get transferred to
            'screen_width': 64,     # width of rendered images
            'screen_height': 64,    # height of rendered images
            'env_name': None,   # name of openai/gym environment
            'reward_norm': 1.,  # reward normalization factor
            'punish_reward': -100,   # reward used when action leads to simulation crash
            'unwrap_time': True,    # removes time limit wrapper from envs so that done is not set on timeout
        })
        return default_dict

    def _postprocess_info(self, info):
        """Sorts solved subtasks into separately logged elements."""
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






class Maze:
    """
    Physics simulation of a PointMaze environment. Credits to SalesForce.
    """

    def __init__(self, *segment_dicts, goal_squares=None, start_squares=None):
        self.pos, self.goal = None, None
        self._segments = {'origin': {'loc': (0.0, 0.0), 'connect': set()}}
        self._locs = set()
        self._locs.add(self._segments['origin']['loc'])
        self._walls = set()
        for direction in ['up', 'down', 'left', 'right']:
            self._walls.add(self._wall_line(self._segments['origin']['loc'], direction))
        self._last_segment = 'origin'
        self.canvas = None

        if goal_squares is None:
            self._goal_squares = None
        elif isinstance(goal_squares, str):
            self._goal_squares = [goal_squares.lower()]
        elif isinstance(goal_squares, list):
            self._goal_squares = goal_squares
        else:
            raise TypeError

        if start_squares is None:
            self.start_squares = ['origin']
        elif isinstance(goal_squares, str):
            self.start_squares = [start_squares.lower()]
        elif isinstance(goal_squares, list):
            self.start_squares = start_squares
        else:
            raise TypeError

        for segment_dict in segment_dicts:
            self._add_segment(**segment_dict)
        self._finalize()

    @staticmethod
    def _wall_line(coord, direction):
        x, y = coord
        if direction == 'up':
            w = [(x - 0.5, x + 0.5), (y + 0.5, y + 0.5)]
        elif direction == 'right':
            w = [(x + 0.5, x + 0.5), (y + 0.5, y - 0.5)]
        elif direction == 'down':
            w = [(x - 0.5, x + 0.5), (y - 0.5, y - 0.5)]
        elif direction == 'left':
            w = [(x - 0.5, x - 0.5), (y - 0.5, y + 0.5)]
        else:
            raise ValueError
        w = tuple([tuple(sorted(line)) for line in w])
        return w

    def _add_segment(self, name, anchor, direction, connect=None, times=1):
        name = str(name).lower()
        original_name = str(name).lower()
        if times > 1:
            assert connect is None
            last_name = str(anchor).lower()
            for time in range(times):
                this_name = original_name + str(time)
                self._add_segment(name=this_name.lower(), anchor=last_name, direction=direction)
                last_name = str(this_name)
            return

        anchor = str(anchor).lower()
        assert anchor in self._segments
        direction = str(direction).lower()

        final_connect = set()

        if connect is not None:
            if isinstance(connect, str):
                connect = str(connect).lower()
                assert connect in ['up', 'down', 'left', 'right']
                final_connect.add(connect)
            elif isinstance(connect, (tuple, list)):
                for connect_direction in connect:
                    connect_direction = str(connect_direction).lower()
                    assert connect_direction in ['up', 'down', 'left', 'right']
                    final_connect.add(connect_direction)

        sx, sy = self._segments[anchor]['loc']
        dx, dy = 0.0, 0.0
        if direction == 'left':
            dx -= 1
            final_connect.add('right')
        elif direction == 'right':
            dx += 1
            final_connect.add('left')
        elif direction == 'up':
            dy += 1
            final_connect.add('down')
        elif direction == 'down':
            dy -= 1
            final_connect.add('up')
        else:
            raise ValueError

        new_loc = (sx + dx, sy + dy)
        assert new_loc not in self._locs

        self._segments[name] = {'loc': new_loc, 'connect': final_connect}
        for direction in ['up', 'down', 'left', 'right']:
            self._walls.add(self._wall_line(new_loc, direction))
        self._locs.add(new_loc)

        self._last_segment = name

    def _finalize(self):
        for segment in self._segments.values():
            n = segment['loc']
            adjacents = {'right': (n[0]+1, n[1]),
                         'left': (n[0]-1, n[1]),
                         'up': (n[0], n[1]+1),
                         'down': (n[0], n[1]-1)}
            segment['connect'] = [k for k, v in adjacents.items() if v in self._locs]
            for c_dir in list(segment['connect']):
                wall = self._wall_line(segment['loc'], c_dir)
                if wall in self._walls:
                    self._walls.remove(wall)

        if self._goal_squares is None:
            self.goal_squares = [self._last_segment]
        else:
            self.goal_squares = []
            for gs in self._goal_squares:
                assert gs in self._segments
                self.goal_squares.append(gs)

    def sample_start(self):
        min_wall_dist = 0.05

        s_square = self.start_squares[np.random.randint(low=0, high=len(self.start_squares))]
        s_square_loc = self._segments[s_square]['loc']

        while True:
            shift = np.random.uniform(low=-0.5, high=0.5, size=(2,))
            loc = s_square_loc + shift
            dist_checker = np.array([min_wall_dist, min_wall_dist]) * np.sign(shift)
            stopped_loc = self.move(loc, dist_checker)
            if float(np.sum(np.abs((loc + dist_checker) - stopped_loc))) == 0.0:
                break
        self.pos = loc[0], loc[1]
        return loc[0], loc[1]

    def sample_goal(self, min_wall_dist=None):
        g_square = self.goal_squares[np.random.randint(low=0, high=len(self.goal_squares))]
        g_square_loc = self._segments[g_square]['loc']
        self.goal = g_square_loc[0], g_square_loc[1]
        return g_square_loc[0], g_square_loc[1]

    def move(self, coord_start, coord_delta, depth=None):
        if depth is None:
            depth = 0
        cx, cy = coord_start
        loc_x0 = np.round(cx)
        loc_y0 = np.round(cy)
        #assert (float(loc_x0), float(loc_y0)) in self._locs
        dx, dy = coord_delta
        loc_x1 = np.round(cx + dx)
        loc_y1 = np.round(cy + dy)
        d_loc_x = int(np.abs(loc_x1 - loc_x0))
        d_loc_y = int(np.abs(loc_y1 - loc_y0))
        xs_crossed = [loc_x0 + (np.sign(dx) * (i + 0.5)) for i in range(d_loc_x)]
        ys_crossed = [loc_y0 + (np.sign(dy) * (i + 0.5)) for i in range(d_loc_y)]

        rds = []

        for x in xs_crossed:
            r = (x - cx) / dx
            loc_x = np.round(cx + (0.999 * r * dx))
            loc_y = np.round(cy + (0.999 * r * dy))
            direction = 'right' if dx > 0 else 'left'
            crossed_line = self._wall_line((loc_x, loc_y), direction)
            if crossed_line in self._walls:
                rds.append([r, direction])

        for y in ys_crossed:
            r = (y - cy) / dy
            loc_x = np.round(cx + (0.999 * r * dx))
            loc_y = np.round(cy + (0.999 * r * dy))
            direction = 'up' if dy > 0 else 'down'
            crossed_line = self._wall_line((loc_x, loc_y), direction)
            if crossed_line in self._walls:
                rds.append([r, direction])

        # The wall will only stop the agent in the direction perpendicular to the wall
        if rds:
            rds = sorted(rds)
            r, direction = rds[0]
            if depth < 3:
                new_dx = r * dx
                new_dy = r * dy
                repulsion = float(np.abs(np.random.rand() * 0.01))
                if direction in ['right', 'left']:
                    new_dx -= np.sign(dx) * repulsion
                    partial_coords = cx + new_dx, cy + new_dy
                    remaining_delta = (0.0, (1 - r) * dy)
                else:
                    new_dy -= np.sign(dy) * repulsion
                    partial_coords = cx + new_dx, cy + new_dy
                    remaining_delta = ((1 - r) * dx, 0.0)
                return self.move(partial_coords, remaining_delta, depth+1)
        else:
            r = 1.0

        dx *= r
        dy *= r
        self.pos = cx + dx, cy + dy
        return cx + dx, cy + dy

    def get_canvas(self):
        """ Renders the structure of the maze as an RGB. Automatically adjusts the spacial resolution. """
        canvas = np.zeros((64, 64, 3), dtype=np.float32)

        # Compute size of each cell, as well as padding and positioning
        wall_size = 0
        max_h = int(max([v['loc'][1] for _, v in self._segments.items()]))
        min_h = int(min([v['loc'][1] for _, v in self._segments.items()]))
        h_range = int(max_h - min_h + 1)
        max_w = int(max([v['loc'][0] for _, v in self._segments.items()]))
        min_w = int(min([v['loc'][0] for _, v in self._segments.items()]))
        w_range = int(max_w - min_w + 1)
        cell_size = (64 - wall_size) // max(h_range, w_range)
        cell_size -= wall_size
        w_padding = (64 - ((cell_size + wall_size) * w_range + wall_size)) // 2
        h_padding = (64 - ((cell_size + wall_size) * h_range + wall_size)) // 2

        # Compute a projection from environment coordinates to image coordinates
        to_coord = lambda pos: (int(np.rint(h_padding+wall_size+(pos[1]+0.5-min_h)*(cell_size+wall_size))),
                                int(np.rint(w_padding+wall_size+(pos[0]+0.5-min_w)*(cell_size+wall_size))))

        for _, v in self._segments.items():
            x, y = int(v['loc'][0]), int(v['loc'][1])
            idxs = (*to_coord((x-0.5, y-0.5)), *to_coord((x+0.5, y+0.5)))
            # Draw a single cell
            canvas[idxs[0]:(idxs[2]-wall_size), idxs[1]:(idxs[3]-wall_size)] = 1.
            for d in v['connect']:
                if d == 'left':
                    ridxs = (*to_coord((x-1.5, y)), *to_coord((x-0.5, y)))
                elif d == 'right':
                    ridxs = (*to_coord((x+0.5, y)), *to_coord((x+1.5, y)))
                elif d == 'up':
                    ridxs = (*to_coord((x, y+0.5)), *to_coord((x, y+1.5)))
                elif d == 'down':
                    ridxs = (*to_coord((x, y-1.5)), *to_coord((x, y-0.5)))
                # Draw a rectangle joining the two cells
                canvas[min(ridxs[0], idxs[0]):(max(ridxs[2], idxs[2])-wall_size), min(ridxs[1], idxs[1]):(max(ridxs[3], idxs[3])-wall_size)] = 1.

        return canvas, to_coord

    def render(self, w, h, mode='offscreen', camera_name='corner'):
        if self.canvas is None:
            self.canvas, self.to_coord = self.get_canvas()
        canvas = self.canvas.copy()
        enlarge = lambda pos: [(pos[0] - 1 + (i // 3), pos[1] - 1 + (i % 3)) for i in range(9)]
        clip = lambda l: [np.clip(e, 0, 63) for e in l]
        to_idx = lambda l: (tuple([e[0] for e in l]), tuple([e[1] for e in l]))
        canvas[to_idx(clip(enlarge(self.to_coord(self.pos))))] = [1., 0., 0.]
        canvas[to_idx(clip(enlarge(self.to_coord(self.goal))))] = [0., 1., 0.]
        return np.flip(canvas, 0)


def make_u_maze(corridor_length):
    """ Function creating an u-shaped maze. """
    corridor_length = int(corridor_length)
    assert corridor_length >= 1

    segments = []
    last = 'origin'
    for x in range(1, corridor_length + 1):
        next_name = '0,{}'.format(x)
        segments.append({'anchor': last, 'direction': 'right', 'name': next_name})
        last = str(next_name)

    assert last == '0,{}'.format(corridor_length)

    up_size = 2

    for x in range(1, up_size+1):
        next_name = '{},{}'.format(x, corridor_length)
        segments.append({'anchor': last, 'direction': 'up', 'name': next_name})
        last = str(next_name)

    assert last == '{},{}'.format(up_size, corridor_length)

    for x in range(1, corridor_length + 1):
        next_name = '{},{}'.format(up_size, corridor_length - x)
        segments.append({'anchor': last, 'direction': 'left', 'name': next_name})
        last = str(next_name)

    assert last == '{},0'.format(up_size)

    return Maze(*segments, goal_squares=last)


def make_room_maze(corridor_length):
    segments = []
    added = [(0,0)]
    names = [(x, y) for x in range(1-corridor_length, corridor_length) for y in range(1-corridor_length, corridor_length)] # if not (max(abs(x), abs(y)) == 1 and x*y != 0)]
    names.remove((0,0))
    to_add = None
    while names:
        for n in names:
            adjacents = [((n[0], n[1]+1), 'down', n),
                         ((n[0], n[1]-1), 'up', n),
                         ((n[0]+1, n[1]), 'left', n),
                         ((n[0]-1, n[1]), 'right', n)]
            for a in adjacents:
                if a[0] in added:
                    to_add = a
                    break
            if to_add is not None:
                break
        anchor = f'{to_add[0][0]},{to_add[0][1]}' if to_add[0] != (0,0) else 'origin'
        segments.append({'anchor': anchor, 'direction': to_add[1], 'name': f'{to_add[2][0]},{to_add[2][1]}'})
        added.append(to_add[2])
        names.remove(to_add[2])
        to_add = None

    l = corridor_length-1
    return Maze(*segments, goal_squares=[f'{l},{l}', f'{-l},{l}', f'{l},{-l}', f'{-l},{-l}'])


def make_cross_maze(corridor_length):
    """ Function creating a cross-shaped maze. """
    corridor_length = int(corridor_length)
    assert corridor_length >= 1

    segments = []
    goal_squares = []
    for direction in ['right', 'left', 'up', 'down']:
        last = 'origin'
        for x in range(1, corridor_length + 1):
            next_name = {'right': f'0,{x}', 'left': f'0,{-x}', 'up': f'{x},0', 'down': f'{-x},0'}[direction]
            segments.append({'anchor': last, 'direction': direction, 'name': next_name})
            last = str(next_name)
        goal_squares.append(last)
    return Maze(*segments, goal_squares=goal_squares)


def make_custom_maze(): 
    " Function creating a fixed intricate maze. "   
    segments = [
        {'anchor': 'origin', 'direction': 'right', 'name': '0,1'},
        {'anchor': '0,1', 'direction': 'right', 'name': '0,2'},
        {'anchor': '0,2', 'direction': 'down', 'name': '-1,2'},
        {'anchor': '-1,2', 'direction': 'right', 'name': '-1,3'},
        {'anchor': '-1,3', 'direction': 'right', 'name': '-1,4'},
        {'anchor': 'origin', 'direction': 'down', 'name': '-1,0'},
        {'anchor': '-1,0', 'direction': 'down', 'name': '-2,0'},
        {'anchor': '-2,0', 'direction': 'down', 'name': '-3,0'},
        {'anchor': '-3,0', 'direction': 'right', 'name': '-3,1'},
        {'anchor': '-3,1', 'direction': 'right', 'name': '-3,2'},
        {'anchor': '-3,2', 'direction': 'right', 'name': '-3,3'},
        {'anchor': '-3,3', 'direction': 'right', 'name': '-3,4'},
        {'anchor': '0,1', 'direction': 'up', 'name': '1,1'},
        {'anchor': '1,1', 'direction': 'up', 'name': '2,1'},
        {'anchor': '2,1', 'direction': 'up', 'name': '3,1'},
        {'anchor': '3,1', 'direction': 'up', 'name': '4,1'},
        {'anchor': '2,1', 'direction': 'right', 'name': '2,2'},
        {'anchor': '2,2', 'direction': 'right', 'name': '2,3'},
        {'anchor': '2,3', 'direction': 'right', 'name': '2,4'},
        {'anchor': '2,4', 'direction': 'down', 'name': '1,4'},
        {'anchor': '2,3', 'direction': 'up', 'name': '3,3'},
        {'anchor': '3,3', 'direction': 'up', 'name': '3,4'},
        {'anchor': '2,1', 'direction': 'left', 'name': '2,0'},
        {'anchor': '2,0', 'direction': 'left', 'name': '2,-1'},
        {'anchor': '2,-1', 'direction': 'left', 'name': '2,-2'},
        {'anchor': '2,-2', 'direction': 'left', 'name': '2,-3'},
        {'anchor': '2,-3', 'direction': 'up', 'name': '3,-3'},
        {'anchor': '3,-3', 'direction': 'up', 'name': '4,-3'},
        {'anchor': '4,-3', 'direction': 'right', 'name': '4,-2'},
        {'anchor': '4,-2', 'direction': 'right', 'name': '4,-1'},
        {'anchor': '2,-2', 'direction': 'down', 'name': '1,-2'},
        {'anchor': '1,-2', 'direction': 'down', 'name': '0,-2'},
        {'anchor': '0,-2', 'direction': 'down', 'name': '-1,-2'},
        {'anchor': '-1,-2', 'direction': 'down', 'name': '-2,-2'},
    ]
    return Maze(*segments, goal_squares=['-3,4', '1,4', '4,1', '4,-1', '-2,-2'], start_squares=['-1,4'])
 
def make_rebuttal_maze(length): 
    " Function creating a fixed intricate maze. "

    def get_coords(last_pos):
        if last_pos == 'origin':
            last_pos = '0,0'
        x, y = last_pos.split(',')
        x, y = int(x), int(y)
        return x, y
    
    def distance(pos1, pos2):
        x1, y1 = get_coords(pos1)
        x2, y2 = get_coords(pos2)
        return max(np.abs(x1-x2), np.abs(y1-y2))
    
    def reverse(candidate):
        rd = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
        rc = {'anchor': candidate['name'], 'name': candidate['anchor'], 'direction': rd[candidate['direction']]}

    def get_name(last_pos, dir):
        x, y = get_coords(last_pos)
        x_offset = {'u': 0, 'd': 0, 'r': 1, 'l': -1}
        y_offset = {'u': 1, 'd': -1, 'r': 0, 'l': 0}
        x, y = x+x_offset[dir], y+y_offset[dir]
        return f'{x},{y}'
    
    ext = {'l': 'left', 'r': 'right', 'u': 'up', 'd': 'down'}
    paths = ['rruulll', 'rrd', 'rullddrr']
    segments = []
    for p in paths:
        last_pos = 'origin'
        for dir in p:
            for i in range(length):
                new_pos = get_name(last_pos, dir)
                new_segment = {'anchor': last_pos, 'direction': ext[dir], 'name': new_pos}
                if new_segment not in segments:
                    segments.append(new_segment)
                last_pos = new_pos
    
    '''
    cells = [s['anchor'] for s in segments]
    for cell in cells:
        print(cell)
        queue = [cell]
        while queue:
            current_cell = queue.pop()
            candidates = [{'anchor': current_cell, 'direction': ext[d], 'name': get_name(current_cell, d)} for d in 'udlr']
            for candidate in candidates:
                if distance(cell, candidate['name']) > length // 2:
                    continue
                if candidate in segments or reverse(candidate) in segments:
                    continue
                queue.append(candidate['name'])
                segments.append(candidate)
    '''
    
    return Maze(*segments, goal_squares=[f'{-length},{2*length}'], start_squares=[f'{-length},{-length}'])



# remember normalization!