from collections import namedtuple
import random

from ...bayesian import Normal, Dirichlet
from ...mdp import MDP, value_iteration, policy_iteration
from .core import Algorithm
from ..environment import initial_value
from .posteriorsampling import *

class NValueSampling(PosteriorSampling):
    def __init__(self, env, p_reward=lambda: Normal(0, 1, 1), n=2):
        self.env = env
        self.sampler, self.posterior = create_sampler(env, p_reward)
        self.n = n

        self._updated_policy = False

    def init_episode(self):
        mdp_samples = [value_iteration(self.sampler(), epsilon=1e-3) for _ in range(self.n)]
        policy_values = [(pair[1], initial_value(self.env, pair[0])) for pair in mdp_samples]
        self.policy = max(policy_values, key=lambda x: x[1])[0]

    def __repr__(self):
        return "N-Value Sampling"


class TwoMDPSampling(PosteriorSampling):
    def __init__(self, env, p_reward=lambda: Normal(0, 1, 1), thr=0.3):
        self.env = env
        self.sampler, self.posterior = create_sampler(env, p_reward)
        self.thr = thr

        self._updated_policy = False

    def init_episode(self, max_tries=15):
    
        tries = 0
        while tries < max_tries:
            mdp1 = self.sampler()
            mdp2 = self.sampler()
            values11, policy1 = value_iteration(mdp1, epsilon=1e-3)
            values22, policy2 = value_iteration(mdp2, epsilon=1e-3)        
            values12 = policy_iteration(mdp2, policy1, epsilon=1e-3, values=values22)
            values21 = policy_iteration(mdp1, policy2, epsilon=1e-3, values=values11)
            tries += 1

            z1 = initial_value(self.env, values11) - initial_value(self.env, values21)
            z2 = initial_value(self.env, values22) - initial_value(self.env, values12)
            z = z1 / initial_value(self.env, values11) + z2 / initial_value(self.env, values22)
            if z >= 2 * self.thr:
                self.policy = policy1 if random.random() < 0.5 else policy2
                return

        self.policy = policy1

    def __repr__(self):
        return "TwoMDPValue Sampling"
