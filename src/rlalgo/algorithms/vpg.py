from ..algorithm import Algorithm
from ..policy import PolicyGradientPolicy
from ..env import Env
from ..log import logging, log
import torch as th
from torch import Tensor
from dataclasses import dataclass
from typing import override

@dataclass
class VPG(Algorithm[PolicyGradientPolicy]):
    gamma: float = 0.99
    epochs: int = 1
    lr: float = 0.001

    @override
    def train(self, model: PolicyGradientPolicy, env: Env) -> None:
        opt = th.optim.Adam(model.parameters(), self.lr)
        with logging('epoch', mode='plot'):
            for _ in range(self.epochs):
                opt.zero_grad()
                self.trajectory(model, env).backward()
                opt.step()

    @staticmethod
    def trajectory(model: PolicyGradientPolicy, env: Env) -> Tensor:
        obss = []; acts = []; rews = []
        obs = env.reset()
        while True:
            act = model.act(obs)
            obss.append(obs); acts.append(act)
            obs, rew, term, trunc = env.step(act)
            rews.append(rew)
            if term or trunc:
                break
        obss, acts, rews = [th.stack(l) for l in (obss, acts, rews)]
        logps = model.log_prob(obss, acts)
        gs = rews.flip(0).cumsum(0).flip(0)
        loss = th.mean(logps * gs)
        log(r'$\bar{r}_\tau$', rews.sum(dim=0).mean().item())
        return loss

