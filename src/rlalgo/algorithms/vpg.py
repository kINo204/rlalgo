from ..algorithm import Algorithm
from ..policy import PolicyGradientPolicy
from ..env import Env
import torch as th
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
        for _ in range(self.epochs):
            obss = []; acts = []; rews = []
            obs = env.reset()
            while True:
                act = model.act(obs)
                obss.append(obs); acts.append(act)
                obs, rew, term, trunc = env.step(act)
                rews.append(rew)
                if term.any() or trunc.any():
                    break
            obss, acts, rews = [th.stack(l) for l in (obss, acts, rews)]
            logps = model.log_prob(obss, acts)
            gs = rews.flip(0).cumsum(0).flip(0)
            opt.zero_grad()
            loss = th.mean(logps * gs)
            loss.backward()
            opt.step()
