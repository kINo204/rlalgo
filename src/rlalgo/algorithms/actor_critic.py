from typing import Callable, override

import torch as th
import torch.nn.functional as F
from torch import Tensor

from ..algorithm import PolicyGradientAlgo
from ..env import Env
from ..log import log, logging
from ..policy import ActorCriticPolicy


class ActorCritic(PolicyGradientAlgo[ActorCriticPolicy]):
    def __init__(self,
                 stop_threshold: int = 480,
                 stop_window: int = 25,
                 stop_tolerance: int = 5,
                 epochs: int = 120,
                 lr: float = 0.006,
                 gamma: float = 0.99,
                 alpha: float = 1.2,
                 decay: float = 0.95,
                 decay_threshold: float = 300,
                 decay_step: int = 10,
                 ) -> None:
        super().__init__(epochs, stop_threshold, stop_window, stop_tolerance, lr)
        self.gamma = gamma
        self.alpha = alpha
        self.decay = decay
        self.decay_thr = decay_threshold
        self.decay_step = decay_step

    @override
    def run(self, policy: ActorCriticPolicy, env: Env, optstep: Callable[[Tensor], None]) -> None:
        '''Overriding for `alpha` exponential decay, stablizing a good value estimator.'''
        with logging('epoch', mode='plot'):
            cnt: int = 0
            e = 0
            while True:
                e += 1
                loss, mean_rew = self.trajectory(policy, env)
                optstep(loss)
                print('epoch:', e, 'mean-reward:', mean_rew)
                log(r'$\bar{r}(\tau)$', mean_rew.item())
                log(r'$\mathcal{L}(\tau)$', loss.item())

                cnt = (cnt + 1) % self.decay_step
                if cnt == 0 and mean_rew > self.decay_thr:
                    self.alpha = max(self.decay * self.alpha, 0.8)
                if self.stopper.step(mean_rew):
                    break

    @override
    def criterion(self, policy: ActorCriticPolicy, obss: Tensor, acts: Tensor, rews: Tensor) -> Tensor:
        vs = policy.value(obss)
        with th.no_grad():
            tds = rews + self.gamma * vs[1:] - vs[:-1]
        logps = policy.log_prob(obss[:-1], acts)
        loss_actor = - th.mean(logps * tds)
        loss_critic = F.mse_loss(vs[:-1], rews + self.gamma * vs[1:])
        loss = loss_actor + self.alpha * loss_critic
        return loss
