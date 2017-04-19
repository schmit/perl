import random
import statistics
import numpy as np
import copy

from ...priors import NormalPrior
from ..environment import mdp_to_env, env_value
from ..simulator import episode
from ...mdp import policy_iteration, value_iteration, imp_value_iteration
from .posteriorsampling import PosteriorSampling, create_sampler, update_posteriors, Posterior, sample_mdp, Indexer
from .core import FixedPolicy
from ...util import sample, sars

class InfoImprovSampling(PosteriorSampling):
    def __init__(self,
            mdp,
            p_reward=lambda: NormalPrior(0, 1, 1),
            discount=0.95,
            q=4.0,
            k=10,
            total_budget=100,
            num_epis_data=5,
            alpha=2,
            verbose=False):

        self.env = mdp_to_env(mdp)
        self.sampler, self.posterior = create_sampler(
                mdp,
                self.env,
                p_reward,
                discount)

        self.num_policies = q
        self.num_mdps = k
        self.num_epis_data = num_epis_data
        self.alpha = alpha
        self.total_budget = total_budget

        self.verbose = verbose

        self.latest_values = None
        self._updated_policy = False
        self._training_logs = []

        self.seen_episodes = 0

    def init_episode(self):

        # v1, v2 = self.compute_entropy(self.posterior, eps=0.005)
        # print("{} | IM Entropy = {} | Mean Val = {}.".format(self.seen_episodes, v1, v2))

        if np.random.random() < self.seen_episodes / self.total_budget:
            values, policy = imp_value_iteration(self.sampler(), epsilon=1e-3)
            self.policy = policy
            self.seen_episodes += 1
            return

        # candidate policies
        policy_set = []
        for i in range(self.num_policies):
            mdp = self.sampler()
            value, policy = imp_value_iteration(mdp, epsilon=1e-3) # , values=self.latest_values)
            self.latest_values = value
            policy_set.append(policy)

        # mdp's to generate data
        mdp_set = [self.sampler() for _ in range(self.num_mdps)]

        p_postvalue = [] 
        for i, pol in enumerate(policy_set):
            score_pol = 0
            for j, m in enumerate(mdp_set):
                posterior_v = self.copy_posterior(self.posterior)
                # generate data using (m, pol) and update posterior
                posterior_v = self.update_posterior_after_sampling(posterior_v, pol, m, self.num_epis_data)
                # compute score of the new posterior
                score = self.compute_score(posterior_v)
                score_pol += score
            score_pol /= self.num_mdps
            p_postvalue.append(score_pol)

        p_star = np.argmax(p_postvalue)
        self._training_logs.append(p_postvalue)

        if self.verbose:
            print("{} | ValueUppBounds: {}".format(self.seen_episodes, p_postvalue))
            print("Policy: {}".format(policy_set[p_star]))

        self.policy = policy_set[p_star]
        self.seen_episodes += 1

        return None

    def copy_posterior(self, posterior):
        # returns a copy
        t = copy.deepcopy(posterior.transitions)
        r = copy.deepcopy(posterior.rewards)
        from_index = copy.deepcopy(posterior.indexer.from_index)
        to_index = copy.deepcopy(posterior.indexer.to_index)
        i = Indexer(from_index, to_index)
        return Posterior(t, r, i)

    def compute_score(self, mdp_posterior, total_samples=20):
        # compute the score of V(mdp, pol), mdp and pol are indep samples from mdp_posterior

        cross_values = []

        for i in range(total_samples):
            mdp = sample_mdp(self.env, mdp_posterior, self.env.discount)
            mdp2 = sample_mdp(self.env, mdp_posterior, self.env.discount)
            values, policy = imp_value_iteration(mdp2, epsilon=1e-3, values=self.latest_values)
            cross_v = env_value(self.env, policy_iteration(mdp, policy, values=values))
            cross_values.append(cross_v)

        # print("Tries {} | entropy_est = {}.".format(tries, entropy_est[-1]))
        return np.mean(cross_values) + self.alpha * np.std(cross_values)

    def update_posterior_after_sampling(self, posterior, policy_s, mdp_s, num_episodes=5):

        env_s = mdp_to_env(mdp_s)
        algo_s = FixedPolicy(policy_s)
        data = []

        for epi in range(num_episodes):
            # run an episode
            history = episode(env_s, algo_s)
            steps = sars(history)
            data += steps
        
        update_posteriors(data, posterior)
        return posterior

    def __repr__(self):
        return "InfoImprov-Sampling"