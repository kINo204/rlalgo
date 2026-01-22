from ..algorithm import Algorithm
from ..policy import ActorCriticPolicy
from ..env import Env
from ..log import logging, log
from ..utils import rollout
import torch as th
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import override

@dataclass
class ActorCritic(Algorithm[ActorCriticPolicy]):
    gamma: float = 0.99
    alpha: float = 0.85
    epochs: int = 1000
    stop_window: int = 20
    lr: float = 0.008

    @override
    def train(self, model: ActorCriticPolicy, env: Env) -> None:
        opt = th.optim.Adam(model.parameters(), self.lr)
        with logging('epoch', mode='plot'):
            window = self.stop_window
            for e in range(self.epochs):
                loss, mean_rew = self.trajectory(model, env)
                print('epoch:', e, 'mean-reward:', mean_rew)
                log(r'$\bar{r}(\tau)$', mean_rew.item())
                log(r'$\mathcal{L}(\tau)$', loss.item())

                if mean_rew < 450:
                    window = self.stop_window
                else:
                    window -= 1
                    if window == 0: break

                opt.zero_grad()
                loss.backward()
                opt.step()

    def trajectory(self, model: ActorCriticPolicy, env: Env) -> tuple[Tensor, Tensor]:
        obss, acts, rews = [th.stack(l) for l in rollout(model, env)]
        vs = model.value(obss)
        with th.no_grad():
            tds = rews + self.gamma * vs[1:] - vs[:-1]
        logps = model.log_prob(obss, acts)
        loss_actor = - th.mean(logps * tds)
        loss_critic = F.mse_loss(vs[:-1], rews + self.gamma * vs[1:])
        loss = loss_actor + self.alpha * loss_critic
        mean_rew = rews.sum(dim=0).mean()
        return loss, mean_rew