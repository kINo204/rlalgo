from rlalgo import Policy, Algorithm, Env, PolicyGradientPolicy
from rlalgo.env import GymEnv
from rlalgo.algorithms import VPG
import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

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

    def act(self, obs: Tensor) -> Tensor:
        with th.no_grad():
            logits: Tensor = self(obs)
            action = logits.argmax(dim=-1).unsqueeze(dim=-1)
            return action

    def log_prob(self, obs: Tensor, act: Tensor) -> Tensor:
        logits: Tensor = self(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=act).squeeze()


policy: Policy = VPNPolicy()
algo: Algorithm = VPG(epochs=100)
env: Env = GymEnv("CartPole-v1", nenvs=2)

algo.train(policy, env)
