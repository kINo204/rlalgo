import rlalgo as rl
from rlalgo import Policy, Algorithm, Env
from rlalgo.policy import ActorCriticPolicy
from rlalgo.env import GymEnv
from rlalgo.algorithms import ActorCritic
import torch as th
from torch import nn, Tensor
from torch.distributions import Categorical
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation as NormObs
from typing import override

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


policy: Policy = ACPolicy()
algo: Algorithm = ActorCritic()
env: Env = GymEnv(NormObs(gym.make('CartPole-v1')))

algo.train(policy, env)

env: Env = GymEnv(NormObs(gym.make('CartPole-v1', render_mode='human')))
rl.rollout(policy, env)