from ..algorithm import Algorithm
from ..policy import PolicyGradientPolicy
from ..env import Env
from ..log import logging, log
from ..utils import rollout
import torch as th
from torch import Tensor
from dataclasses import dataclass
from typing import override

@dataclass
class VPG(Algorithm[PolicyGradientPolicy]):
    gamma: float = 0.99
    epochs: int = 1000
    stop_window: int = 20
    lr: float = 0.006

    @override
    def train(self, model: PolicyGradientPolicy, env: Env) -> None:
        opt = th.optim.Adam(model.parameters(), self.lr)
        with logging('epoch', mode='plot'):
            window = self.stop_window
            for e in range(self.epochs):
                loss, mean_rew = self.trajectory(model, env)
                print('epoch:', e, 'mean-reward:', mean_rew)
                log(r'$\bar{r}(\tau)$', mean_rew.item())
                log(r'$\mathcal{L}(\tau)$', loss.item())

                if mean_rew < 480:
                    window = self.stop_window
                else:
                    window -= 1
                    if window == 0: break

                opt.zero_grad()
                loss.backward()
                opt.step()

    @staticmethod
    def trajectory(model: PolicyGradientPolicy, env: Env) -> tuple[Tensor, Tensor]:
        obss, acts, rews = [th.stack(l) for l in rollout(model, env)]
        logps = model.log_prob(obss, acts)
        gs = rews.flip(0).cumsum(0).flip(0)
        gs = (gs - gs.mean()) / gs.std() # Key step!
        loss = - th.mean(logps * gs)
        mean_rew = rews.sum(dim=0).mean()
        return loss, mean_rew
