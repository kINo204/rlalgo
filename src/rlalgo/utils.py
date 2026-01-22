from .policy import Policy
from .env import Env
from typing import Any

def rollout[ObsT, ActT, RewT](policy: Policy[ObsT, ActT],
                              env: Env[ObsT, ActT, RewT, Any]) \
        -> tuple[list[ObsT], list[ActT], list[RewT]]:
    obss: list[ObsT] = []
    acts: list[ActT] = []
    rews: list[RewT] = []
    obs = env.reset()
    obss.append(obs)
    while True:
        act = policy.act(obs)
        acts.append(act)
        obs, rew, term, trunc = env.step(act)
        obss.append(obs)
        rews.append(rew)
        if term or trunc:
            break
    return obss, acts, rews