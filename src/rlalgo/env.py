from abc import abstractmethod
from typing import Protocol, override

import numpy as np
import torch as th
from torch import Tensor


class Env[ObsT, ActT, RewT, ArrT](Protocol):
    @abstractmethod
    def reset(self) -> ObsT:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActT) -> tuple[ObsT, RewT, ArrT, ArrT]:
        raise NotImplementedError

class GymEnv(Env):
    '''
    Wrapper of Gymnasium (vectorized) environments.
    '''
    import gymnasium as gym

    env: gym.Env | gym.vector.VectorEnv

    def __init__(self, env: gym.Env | gym.vector.VectorEnv) -> None:
        self.env = env

    @override
    def reset(self) -> Tensor:
        obs, _info = self.env.reset()
        return th.as_tensor(obs)

    @override
    def step(self, action: Tensor) -> tuple[Tensor, Tensor, bool, bool]:
        a = action.to(th.int32) \
            .numpy().reshape(self.env.action_space.shape)
        obs, rew, term, trunc, _info = self.env.step(a)
        obs = th.as_tensor(obs)
        rew = th.as_tensor(rew)
        # TODO vecenv truncation
        if type(term) is np.ndarray:
            term = any(term)
        if type(trunc) is np.ndarray:
            trunc = any(trunc)
        return obs, rew, term, trunc