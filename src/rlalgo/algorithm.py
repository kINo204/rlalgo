from abc import ABC, abstractmethod
from typing import Callable, Protocol, override

import torch as th
from torch import Tensor

from .env import Env
from .log import log, logging
from .policy import Policy, PolicyGradientPolicy
from .util import SlideWindowStopper, rollout


class Algorithm[PolicyT](Protocol):
    @abstractmethod
    def train(self, policy: PolicyT, env: Env) -> None:
        raise NotImplementedError

class OnPolicyAlgorithm[PolicyT: Policy](Algorithm[PolicyT], ABC):
    epochs: int
    stopper: SlideWindowStopper[Tensor]

    def __init__(self,
                 epochs: int,
                 stop_threshold: float,
                 stop_window: int,
                 stop_tolerance: int
                 ) -> None:
        self.epochs = epochs # reserved for non-early-stopping usage
        self.stopper = SlideWindowStopper(lambda m: m.item() > stop_threshold,
                                          stop_window,
                                          stop_tolerance,
                                          min_steps=epochs)

    def run(self, policy: PolicyT, env: Env, optstep: Callable[[Tensor], None]) -> None:
        '''
        A possible implementation for training. Run `epochs` trajectory, perform
        `optstep` on each, and use a sliding window for early-stopping.
        '''
        with logging('epoch', mode='plot'):
            e = 0
            while True:
                e += 1
                loss, mean_rew = self.trajectory(policy, env)
                print('epoch:', e, 'mean-reward:', mean_rew)
                log(r'$\bar{r}(\tau)$', mean_rew.item())
                log(r'$\mathcal{L}(\tau)$', loss.item())
                optstep(loss)
                if self.stopper.step(mean_rew):
                    break

    def trajectory(self, policy: PolicyT, env: Env):
        obss, acts, rews = [th.stack(l) for l in rollout(policy, env)]
        mean_rew = rews.sum(dim=0).mean()
        loss = self.criterion(policy, obss, acts, rews)
        return loss, mean_rew

    @abstractmethod
    def criterion(self, policy: PolicyT, obss: Tensor, acts: Tensor, rews: Tensor) -> Tensor:
        raise NotImplementedError

class PolicyGradientAlgo[PolicyT: PolicyGradientPolicy](OnPolicyAlgorithm[PolicyT], ABC):
    lr: float

    def __init__(self,
                 epochs: int,
                 stop_threshold: float,
                 stop_window: int,
                 stop_tolerance: int,
                 lr: float,
                 ) -> None:
        super().__init__(epochs, stop_threshold, stop_window, stop_tolerance)
        self.lr = lr

    @override
    def train(self, policy: PolicyT, env: Env) -> None:
        opt = th.optim.Adam(policy.parameters(), self.lr)
        def optstep(loss):
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.run(policy, env, optstep)
