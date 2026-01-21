import numpy as np
import gymnasium as gym
import torch as th
from torch import Tensor
from typing import Protocol, override
from abc import abstractmethod

class Env[ObsT, ActT, RewT, ArrT](Protocol):
    @abstractmethod
    def reset(self) -> ObsT:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActT) -> tuple[ObsT, RewT, ArrT, ArrT]:
        raise NotImplementedError

class GymEnv(Env):
    env: gym.Env | gym.vector.VectorEnv

    def __init__(self, envid: str, nenvs: int = 1, **kwargs) -> None:
        if nenvs == 1:
            self.env = gym.make(envid, **kwargs)
        else:
            self.env = gym.make_vec(envid, nenvs)

    @override
    def reset(self) -> Tensor:
        obs, _info = self.env.reset()
        return th.as_tensor(obs)

    @override
    def step(self, action: Tensor) -> tuple[Tensor, Tensor, bool | np.ndarray, bool | np.ndarray]:
        a = action.to(th.int32).reshape(*self.env.action_space.shape).numpy() # pyright: ignore[reportOptionalIterable]
        obs, rew, term, trunc, _info = self.env.step(a)
        obs = th.as_tensor(obs)
        rew = th.as_tensor(rew)
        return obs, rew, term, trunc