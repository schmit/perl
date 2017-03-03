import random
import statistics
import numpy as np

from ...priors import NormalPrior
from ..environment import mdp_to_env, env_value
from ..memory import RingBuffer
from ...mdp import policy_iteration, value_iteration
from .posteriorsampling import PosteriorSampling, create_sampler

class TwoStepPosteriorSampling(PosteriorSampling):
    def __init__(self,
            mdp,
            p_reward=lambda: NormalPrior(0, 1, 1),
            discount=0.95,
            threshold=1.0,
            capacity=10):

        self.env = mdp_to_env(mdp)
        self.sampler, self.posterior = create_sampler(
                mdp,
                self.env,
                p_reward,
                discount)

        self.threshold = threshold
        self.buffer = RingBuffer(capacity)

        self._updated_policy = False

    def init_episode(self, max_tries=15):
        mdp_1 = self.sampler()
        value_1, policy_1 = value_iteration(mdp_1, epsilon=1e-3)

        # half the time, use the first policy
        if random.random() < 0.5:
            self.policy = policy_1
            return None

        # otherwise, find alternative policy
        opt_value_1 = env_value(self.env, value_1)
        while True:
            mdp_2 = self.sampler()
            value_2, policy_2 = value_iteration(mdp_2, epsilon=1e-3)

            opt_value_2 = env_value(self.env, value_2)

            alt_value_1 = env_value(self.env, policy_iteration(mdp_1, policy_2))
            alt_value_2 = env_value(self.env, policy_iteration(mdp_2, policy_1))

            z = opt_value_1 - alt_value_1 + opt_value_2 - alt_value_2
            if z < 0:
                print("Warning TSPS: z < 0")

            if not self.buffer.length or z > max(self.buffer, default=-1e16) or z >= self.threshold * statistics.median(self.buffer):
                self.buffer.add(z)
                self.policy = policy_2
                return None

            # add Z score to buffer
            self.buffer.add(z)


    def __repr__(self):
        return "Two Step Posterior Sampling"


class TwoStepDecoupledPosteriorSampling(PosteriorSampling):
    def __init__(self,
            mdp,
            p_reward=lambda: NormalPrior(0, 1, 1),
            discount=0.95,
            threshold=1.0,
            capacity=10):

        self.env = mdp_to_env(mdp)
        self.sampler, self.posterior = create_sampler(
                mdp,
                self.env,
                p_reward,
                discount)

        self.threshold = threshold
        self.buffer = RingBuffer(capacity)

        self.latest_values = None
        self._updated_policy = False

    def init_episode(self, max_tries=10):

        # half the time, use the first policy
        if random.random() < 0.5:
            mdp = self.sampler()
            value, policy = value_iteration(mdp, epsilon=1e-3, values=self.latest_values)
            self.policy = policy
            self.latest_values = value
            return None

        num_tries = 0
        # otherwise, find alternative policy
        while num_tries < max_tries:
            mdp_1 = self.sampler()
            mdp_2 = self.sampler()
            value_1, policy_1 = value_iteration(mdp_1, epsilon=1e-3, values=self.latest_values)
            value_2, policy_2 = value_iteration(mdp_2, epsilon=1e-3, values=self.latest_values)

            opt_value_1 = env_value(self.env, value_1)
            opt_value_2 = env_value(self.env, value_2)
            alt_value_1 = env_value(self.env, policy_iteration(mdp_1, policy_2))
            alt_value_2 = env_value(self.env, policy_iteration(mdp_2, policy_1))

            z = opt_value_1 - alt_value_1 + opt_value_2 - alt_value_2
            if not self.buffer.length or z > max(self.buffer, default=0) or z >= self.threshold * statistics.median(self.buffer):
                self.buffer.add(z)
                if random.random() < 0.5:
                    self.policy = policy_1
                    self.latest_values = value_1
                else:
                    self.policy = policy_2
                    self.latest_values = value_2
                return None
            num_tries += 1

            # add Z score to buffer
            self.buffer.add(z)

        # if search failed
        print("Search Failed. Computing Random Policy.")
        mdp = self.sampler()
        value, policy = value_iteration(mdp, epsilon=1e-3, values=self.latest_values)
        self.policy = policy
        self.latest_values = value
        return None


    def __repr__(self):
        return "Two Step Decoupled Posterior Sampling"


class MaxVarPosteriorSampling(PosteriorSampling):
    def __init__(self,
            mdp,
            p_reward=lambda: NormalPrior(0, 1, 1),
            discount=0.95,
            q=4.0,
            k=10):

        self.env = mdp_to_env(mdp)
        self.sampler, self.posterior = create_sampler(
                mdp,
                self.env,
                p_reward,
                discount)

        self.num_policies = q
        self.num_mdps = k

        self.latest_values = None
        self._updated_policy = False
        self._training_logs = []

    def init_episode(self):

        policy_set = []
        for i in range(self.num_policies):
            mdp = self.sampler()
            value, policy = value_iteration(mdp, epsilon=1e-3, values=self.latest_values)
            self.latest_values = value
            policy_set.append(policy)

        mdp_set = [self.sampler() for _ in range(self.num_mdps)]

        pol_std = [] 
        for pol in policy_set:
            pol_vals = [env_value(self.env, policy_iteration(mdp, pol)) for mdp in mdp_set]
            pol_std.append(np.std(pol_vals))

        p_star = np.argmax(pol_std)
        self._training_logs.append(pol_std)

        self.policy = policy_set[p_star]
        return None


    def __repr__(self):
        return "VarMax Posterior Sampling"

