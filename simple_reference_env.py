from typing import Any, List, Union, Optional, Dict
import gymnasium as gym
import numpy as np
import pettingzoo
from functools import reduce

from ding.envs import BaseEnv, BaseEnvTimestep, FrameStackWrapper
from ding.torch_utils import to_ndarray, to_list
from ding.envs.common.common_function import affine_transform
from ding.utils import ENV_REGISTRY, import_module
from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe._mpe_utils.simple_env import make_env
from pettingzoo.mpe.simple_reference.simple_reference import raw_env
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import uuid
from pathlib import Path


@ENV_REGISTRY.register("petting_zoo")
class ReferenceEnv(BaseEnv):
    # Now only supports simple_spread_v2.
    # All agents' observations should have the same shape.

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None
        self._env_family = self._cfg.env_family
        self._env_id = self._cfg.env_id
        self._num_agents = self._cfg.n_agent
        self._num_landmarks = self._cfg.n_landmark
        self._continuous_actions = self._cfg.get("continuous_actions", False)
        self._max_cycles = self._cfg.get("max_cycles", 25)
        self._act_scale = self._cfg.get("act_scale", False)
        self._agent_specific_global_state = self._cfg.get(
            "agent_specific_global_state", False
        )
        if self._act_scale:
            assert (
                self._continuous_actions
            ), "Only continuous action space env needs act_scale"

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            _env = make_env(raw_env)
            parallel_env = parallel_wrapper_fn(_env)
            # init env
            self._env = parallel_env(
                continuous_actions=self._continuous_actions,
                max_cycles=self._max_cycles,
            )
        if self._replay_path is not None:
            Path("reference/").mkdir(parents=True, exist_ok=True)
            before_training = f"reference/{str(uuid.uuid4())}.mp4"
            self._video = VideoRecorder(self._env, before_training)
        if hasattr(self, "_seed"):
            obs = self._env.reset(seed=self._seed)
        else:
            obs = self._env.reset()
        if not self._init_flag:
            self._agents = self._env.agents

            self._action_space = gym.spaces.Dict(
                {agent: self._env.action_space(agent) for agent in self._agents}
            )
            single_agent_obs_space = self._env.action_space(self._agents[0])
            if isinstance(single_agent_obs_space, gym.spaces.Box):
                self._action_dim = single_agent_obs_space.shape
            elif isinstance(single_agent_obs_space, gym.spaces.Discrete):
                self._action_dim = (single_agent_obs_space.n,)
            else:
                raise Exception(
                    "Only support `Box` or `Discrete` obs space for single agent."
                )
            # for case when env.agent_obs_only=True
            if not self._cfg.agent_obs_only:
                self._observation_space = gym.spaces.Dict(
                    {
                        "agent_state": gym.spaces.Box(
                            low=float("-inf"),
                            high=float("inf"),
                            shape=(
                                self._num_agents,
                                self._env.observation_space("agent_0").shape[0],
                            ),  # (self._num_agents, 30)
                            dtype=np.float32,
                        ),
                        "global_state": gym.spaces.Box(
                            low=float("-inf"),
                            high=float("inf"),
                            shape=(
                                self._env.observation_space("agent_0").shape[0] + 32
                                if self._agent_specific_global_state
                                else 32,
                            ),
                            dtype=np.float32,
                        ),
                        "action_mask": gym.spaces.Box(
                            low=float("-inf"),
                            high=float("inf"),
                            shape=(
                                self._num_agents,
                                self._action_dim[0],
                            ),  # (self._num_agents, 5)
                            dtype=np.float32,
                        ),
                    }
                )
                # whether use agent_specific_global_state. It is usually used in AC multiagent algos, e.g., mappo, masac, etc.
                if self._agent_specific_global_state:
                    agent_specifig_global_state = gym.spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(
                            self._num_agents,
                            self._env.observation_space("agent_0").shape[0] + 32
                            if self._agent_specific_global_state
                            else 32,
                        ),
                        dtype=np.float32,
                    )
                    self._observation_space[
                        "global_state"
                    ] = agent_specifig_global_state
            else:
                # for case when env.agent_obs_only=True
                self._observation_space = gym.spaces.Box(
                    low=float("-inf"),
                    high=float("inf"),
                    shape=(
                        self._num_agents,
                        self._env.observation_space("agent_0").shape[0],
                    ),
                    dtype=np.float32,
                )

            self._reward_space = gym.spaces.Dict(
                {
                    agent: gym.spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(1,),
                        dtype=np.float32,
                    )
                    for agent in self._agents
                }
            )
            self._init_flag = True
        self._eval_episode_return = 0.0
        self._step_count = 0
        obs_n = self._process_obs(obs)
        return obs_n

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def render(self) -> None:
        self._env.render()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        self._step_count += 1
        assert isinstance(action, np.ndarray), type(action)
        action = self._process_action(action)
        if self._act_scale:
            for agent in self._agents:
                # print(action[agent])
                # print(self.action_space[agent])
                # print(self.action_space[agent].low, self.action_space[agent].high)
                action[agent] = affine_transform(
                    action[agent],
                    min_val=self.action_space[agent].low,
                    max_val=self.action_space[agent].high,
                )

        obs, rew, done, trunc, info = self._env.step(action)
        self.render()
        if self._replay_path is not None:
            self._video.capture_frame()
        obs_n = self._process_obs(obs)
        rew_n = np.array([sum([rew[agent] for agent in self._agents])])
        rew_n = rew_n.astype(np.float32)

        self._eval_episode_return += rew_n.item()

        done_n = (
            reduce(lambda x, y: x and y, done.values())
            or self._step_count >= self._max_cycles
        )

        if done_n:  # or reduce(lambda x, y: x and y, done.values())
            info["eval_episode_return"] = self._eval_episode_return
        return BaseEnvTimestep(obs_n, rew_n, done_n, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = "./video"
        self._replay_path = replay_path

    def _process_action(self, action: "torch.Tensor") -> Dict[str, np.ndarray]:  # noqa
        dict_action = {}
        for i, agent in enumerate(self._agents):
            agent_action = action[i]
            if agent_action.shape == (1,):
                agent_action = agent_action.squeeze()  # 0-dim array
            dict_action[agent] = agent_action
        return dict_action

    def _process_obs(self, obs: "torch.Tensor") -> np.ndarray:  # noqa
        if isinstance(obs, tuple):
            # print("MyObservation1")
            # print(type(obs))
            # print(obs)
            obs = np.array([obs[0][agent] for agent in self._agents]).astype(np.float32)
        else:
            # print("MyObservation2")
            # print(type(obs))
            # print(obs)
            obs = np.array([obs[agent] for agent in self._agents]).astype(np.float32)
        if self._cfg.get("agent_obs_only", False):
            return obs
        ret = {}

        ret["agent_state"] = obs

        ret["global_state"] = np.concatenate(
            [
                obs[0, 2:-11],  # all agents' position + all landmarks' position
                obs[:, :2].flatten(),  # all agents' velocity
                obs[:, -10:].flatten(),  # all agents' communication
            ]
        )

        if self._agent_specific_global_state:
            ret["global_state"] = np.concatenate(
                [
                    ret["agent_state"],
                    np.expand_dims(ret["global_state"], axis=0).repeat(
                        self._num_agents, axis=0
                    ),
                ],
                axis=1,
            )

        ret["action_mask"] = np.ones((self._num_agents, *self._action_dim)).astype(
            np.float32
        )
        return ret

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        for k in random_action:
            if isinstance(random_action[k], np.ndarray):
                pass
            elif isinstance(random_action[k], int):
                random_action[k] = to_ndarray([random_action[k]], dtype=np.int64)
        return random_action

    def __repr__(self) -> str:
        return "DI-engine PettingZoo Env"

    @property
    def agents(self) -> List[str]:
        return self._agents

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space
