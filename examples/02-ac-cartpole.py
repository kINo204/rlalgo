from typing import override

import gymnasium as gym
import torch as th
import torch.nn.functional as F
from gymnasium.wrappers import NormalizeObservation as NormObs
from torch import Tensor, nn
from torch.distributions import Categorical

from rlalgo.algorithms import ActorCritic
from rlalgo.env import GymEnv
from rlalgo.policy import ActorCriticPolicy
from rlalgo.util import rollout


class ACPolicy(ActorCriticPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.fcs = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.actor = nn.Linear(16, 2)
        self.critic = nn.Linear(16, 1)

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.fcs(input)
        return self.actor(feat), self.critic(feat)

    @override
    def act(self, obs: Tensor) -> Tensor:
        with th.no_grad():
            logits: Tensor = self(obs)[0]
            dist = Categorical(logits=logits)
            action = dist.sample().unsqueeze(-1)
            return action

    @override
    def log_prob(self, obs: Tensor, act: Tensor) -> Tensor:
        logits: Tensor = self(obs)[0]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=act).squeeze()

    @override
    def value(self, obs: Tensor) -> Tensor:
        return self(obs)[1].squeeze()


device  = th.device('cpu')

policy = ACPolicy().to_dev(device)
algo = ActorCritic()
env = GymEnv(NormObs(gym.make('CartPole-v1'))).to_dev(device)

algo.train(policy, env)

env = GymEnv(NormObs(gym.make('CartPole-v1', render_mode='human'))).to_dev(device)
rollout(policy, env)