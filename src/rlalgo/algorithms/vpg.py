from ..algorithm import PolicyGradientAlgo
from ..policy import PolicyGradientPolicy
import torch as th
from torch import Tensor
from typing import override

class VPG(PolicyGradientAlgo[PolicyGradientPolicy]):
    def __init__(self,
                 epochs: int = 150,
                 stop_threshold: float = 480.,
                 stop_window: int = 12,
                 stop_tolerance: int = 5,
                 lr: float = 0.006,
                 ) -> None:
        super().__init__(epochs, stop_threshold, stop_window, stop_tolerance, lr)

    @override
    def criterion(self, policy: PolicyGradientPolicy, obss: Tensor, acts: Tensor, rews: Tensor) -> Tensor:
        logps = policy.log_prob(obss[:-1], acts)
        gs = rews.flip(0).cumsum(0).flip(0)
        gs = (gs - gs.mean()) / gs.std() # Key step!
        loss = - th.mean(logps * gs)
        return loss
