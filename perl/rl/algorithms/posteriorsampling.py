from collections import namedtuple
import random

from ...bayesian import Normal, Dirichlet
from ...mdp import MDP, value_iteration
from .core import Algorithm

Posterior = namedtuple("Posterior", "transitions rewards indexer")

def create_prior(all_states, actions, d_reward = lambda: Normal(0, 1, 1)):
    all_states = sorted(all_states)
    nstates = len(all_states)

    transitions = {(state, action): Dirichlet(nstates + 1, 1/(nstates+2))
            for state in all_states for action in actions(state)}

    rewards = {(state, action): d_reward()
            for state in all_states for action in actions(state)}

    indexer = {state: idx for idx, state in enumerate(all_states)}
    indexer[None] = nstates

    return Posterior(transitions, rewards, indexer)

def sample_posterior(posterior):
    transitions = {(state, action): post.sample()
            for (state, action), post in posterior.transitions.items()}
    rewards = {(state, action): post.sample()
            for (state, action), post in posterior.rewards.items()}
    return transitions, rewards

def get_map(posterior):
    transitions = {(state, action): post.map
            for (state, action), post in posterior.transitions.items()}
    rewards = {(state, action): post.map
            for (state, action), post in posterior.rewards.items()}
    return transitions, rewards


def create_sampler(env, d_reward = lambda: Normal(0, 1, 1)):
    """
    Create a sampler to sample MDPs from a posterior

    Args:
        - env: learning environment
        - d_reward: distribution of reward
    """
    # sorted so we can always recover the order from the indexer
    posterior = create_prior(env.states, env.actions, d_reward)
    sampler = lambda: sample_mdp(env, posterior)

    return sampler, posterior


def sample_mdp(env, posterior):
    sampled_transitions, sampled_rewards = sample_posterior(posterior)

    def transitions(state, action):
        # note do not consider the last sampled transition probability
        # as that is the probability to end the MDP
        return [(p, (env.states[i], sampled_rewards[(state, action)]))
                for i, p in enumerate(sampled_transitions[(state, action)][:-1])]

    # make sure discount is less than one because sampled MDP might have
    # infinite cycles
    return MDP(env.initial_states, env.actions, transitions, min(0.97, env.discount))

def map_mdp(env, posterior):
    map_transitions, map_rewards = get_map(posterior)

    def transitions(state, action):
        # note do not consider the last sampled transition probability
        # as that is the probability to end the MDP
        return [(p, (env.states[i], map_rewards[(state, action)]))
                for i, p in enumerate(map_transitions[(state, action)][:-1])]

    # make sure discount is less than one because sampled MDP might have
    # infinite cycles
    return MDP(env.initial_states, env.actions, transitions, min(0.97, env.discount))

def update_posteriors(steps, posterior):
    for state, action, reward, new_state in steps:
        # update transition:
        posterior.transitions[(state, action)].update(posterior.indexer[new_state])
        # update reward
        posterior.rewards[(state, action)].update(reward)


class PosteriorSampling(Algorithm):
    def __init__(self, env, d_reward=lambda: Normal(0, 1, 1)):
        self.env = env
        self.sampler, self.posterior = create_sampler(env, d_reward)

        self._updated_policy = False

    def init_episode(self):
        values, policy = value_iteration(self.sampler(), epsilon=1e-3)
        self.policy = policy

    def act(self, state):
        return self.policy[state]

    def learn(self, steps):
        update_posteriors(steps, self.posterior)
        self._updated_policy = False

    @property
    def optimal_policy(self):
        # cache optimal policy
        if not self._updated_policy:
            value, policy = value_iteration(map_mdp(self.env, self.posterior))
            self._opt_policy = policy
            self._updated_policy = True

        return self._opt_policy

