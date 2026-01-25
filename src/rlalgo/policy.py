from abc import ABC, abstractmethod
from typing import Protocol, Self

from torch import Tensor, device, nn

from rlalgo.device import Dev


class Policy[ObsT, ActT](Dev, Protocol):
    @abstractmethod
    def act(self, obs: ObsT) -> ActT:
        raise NotImplementedError

class Stochastic(Protocol):
    @abstractmethod
    def log_prob(self, obs: Tensor, act: Tensor) -> Tensor:
        raise NotImplementedError

class PolicyGradientPolicy(Policy[Tensor, Tensor], Stochastic, nn.Module, ABC):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def to_dev(self, device: device) -> Self:
        return self.to(device)

class Critic(Protocol):
    @abstractmethod
    def value(self, obs: Tensor) -> Tensor:
        raise NotImplementedError

class ActorCriticPolicy(Critic, PolicyGradientPolicy, ABC):
    def __init__(self) -> None:
        PolicyGradientPolicy.__init__(self)