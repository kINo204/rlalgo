from typing import override

import gymnasium as gym
import torch as th
import torch.nn.functional as F
from gymnasium.wrappers import NormalizeObservation as NormObs
from torch import Tensor, nn
from torch.distributions import Categorical

from rlalgo.algorithms import VPG
from rlalgo.env import GymEnv
from rlalgo.policy import PolicyGradientPolicy
from rlalgo.util import rollout


class VPNPolicy(PolicyGradientPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.fcs = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.fcs(input)

    @override
    def act(self, obs: Tensor) -> Tensor:
        with th.no_grad():
            logits: Tensor = self(obs)
            dist = Categorical(logits=logits)
            action = dist.sample().unsqueeze(-1)
            return action

    @override
    def log_prob(self, obs: Tensor, act: Tensor) -> Tensor:
        logits: Tensor = self(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=act).squeeze()


policy  = VPNPolicy()
algo    = VPG()
env     = GymEnv(NormObs(gym.make('CartPole-v1')))

algo.train(policy, env)

env = GymEnv(NormObs(gym.make('CartPole-v1', render_mode='human')))
rollout(policy, env)
