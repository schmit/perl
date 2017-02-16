import statistics

from ...bayesian import Normal
from ..environment import env_value
from ..memory import RingBuffer
from ...mdp import policy_iteration, value_iteration
from .posteriorsampling import PosteriorSampling, create_sampler


class TwoStepPosteriorSampling(PosteriorSampling):
    def __init__(self, env, p_reward=lambda: Normal(0, 1, 1), threshold=1.0, capacity=10):
        self.env = env
        self.sampler, self.posterior = create_sampler(env, p_reward)
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
            if z > max(self.buffer, default=0) or z >= self.threshold * statistics.median(self.buffer):
                self.buffer.add(z)
                self.policy = policy_2
                return None

            # add Z score to buffer
            self.buffer.add(z)


    def __repr__(self):
        return "Two step Posterior Sampling"
