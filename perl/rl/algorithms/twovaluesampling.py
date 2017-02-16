from collections import namedtuple
import random

from ...bayesian import Normal, Dirichlet
from ...mdp import MDP, value_iteration
from .core import Algorithm
from .posteriorsampling import Posterior, PosteriorSampling

def initial_value(env, values):
    value = 0
    for prob, state in env.initial_states:
        value += prob * values[state]
    return value

class NValueSampling(PosteriorSampling):
    def __init__(self, env, p_reward=lambda: Normal(0, 1, 1), n=2):
        self.env = env
        self.sampler, self.posterior = create_sampler(env, p_reward)
        self.n = n

        self._updated_policy = False

    def init_episode(self):
        mdp_samples = [value_iteration(self.sampler()) for _ in range(self.n)]
        policy_values = [(pair[1], initial_value(self.env, pair[0])) for pair in mdp_samples]
        return max(policy_values, key=lambda x: x[1])[0]

