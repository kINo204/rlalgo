from torch import nn, Tensor
from typing import Protocol
from abc import ABC, abstractmethod

class Policy[ObsT, ActT](Protocol):
    @abstractmethod
    def act(self, obs: ObsT) -> ActT:
        raise NotImplementedError

class Stochastic(Protocol):
    @abstractmethod
    def log_prob(self, obs: Tensor, act: Tensor) -> Tensor:
        raise NotImplementedError

class Critic(Protocol):
    @abstractmethod
    def value(self, obs: Tensor) -> Tensor:
        raise NotImplementedError

class PolicyGradientPolicy(Policy[Tensor, Tensor], Stochastic, nn.Module, ABC):
    def __init__(self) -> None:
        nn.Module.__init__(self)

class ActorCriticPolicy(Critic, PolicyGradientPolicy, ABC):
    def __init__(self) -> None:
        PolicyGradientPolicy.__init__(self)