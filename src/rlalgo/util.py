from .env import Env
from .policy import Policy

class SlideWindowStopper[MetricT]():
    cnt: int
    min_steps: int
    wnd: int
    tol: int
    thr: MetricT
    o_wnd: int
    o_tol: int

    def __init__(self,
                 window: int,
                 threshold: MetricT,
                 tolerance: int,
                 min_steps: int,
                 ) -> None:
        self.o_wnd = self.wnd = window
        self.o_tol = self.tol = tolerance
        self.thr = threshold
        self.cnt = 0
        self.min_steps = min_steps

    def step(self, metric: MetricT) -> bool:
        self.cnt += 1
        if self.cnt < self.min_steps:
            return False
        # elif self.cnt >= 3 * self.min_steps:
        #     return True
        elif metric < self.thr: # type: ignore
            self.tol -= 1
            if self.tol == 0:
                self.wnd = self.o_wnd
        else:
            self.tol = self.o_tol
            self.wnd -= 1
            if self.wnd == 0:
                return True
        return False

def rollout(policy: Policy, env: Env):
    obss = []
    acts = []
    rews = []
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