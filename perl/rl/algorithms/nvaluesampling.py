from collections import namedtuple
import random
import statistics

from ...priors import NormalPrior
from ...mdp import MDP, value_iteration, policy_iteration
from .core import Algorithm
from ..environment import env_value
from ..memory import RingBuffer
from .posteriorsampling import *

class NValueSampling(PosteriorSampling):
    def __init__(self, env, p_reward=lambda: NormalPrior(0, 1, 1), n=2):
        self.env = env
        self.sampler, self.posterior = create_sampler(env, p_reward)
        self.n = n

        self._updated_policy = False

    def init_episode(self):
        mdp_samples = [value_iteration(self.sampler(), epsilon=1e-3) for _ in range(self.n)]
        policy_values = [(pair[1], env_value(self.env, pair[0])) for pair in mdp_samples]
        self.policy = max(policy_values, key=lambda x: x[1])[0]

    def __repr__(self):
        return "N-Value Sampling"

