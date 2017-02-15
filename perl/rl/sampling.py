from collections import namedtuple
import random

from ..bayesian import Normal, Dirichlet
from ..mdp import MDP, find_all_states, value_iteration
from ..simulator import run
from ..util import sars

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

def posterior_sampling(env, episodes, d_reward=lambda: Normal(0, 1, 1), verbose=None):
    sampler, posterior = create_sampler(env, d_reward)
    episode_rewards = []

    cum_rewards = 0
    for episode in range(episodes):
        # start with low accuracy value iteration for efficiency
        # also, after some samples reuse old value for faster value_iteration
        values, policy = value_iteration(sampler(), epsilon=1e-3)

        steps = sars(run(env, policy))
        update_posteriors(steps, posterior)

        # log rewards
        _, _, rewards, _ = zip(*steps)
        total_reward = sum(env.discount**t * reward
                for t, reward in enumerate(rewards))
        cum_rewards += total_reward
        episode_rewards.append(total_reward)

        if verbose and (episode+1) % verbose == 0:
            print("Episode: {}, cumulative reward: {}".format(episode+1, cum_rewards))


    return posterior, episode_rewards


# todo: finish top two sampling
# def top_two_sampling(sampler, accept_condition, beta=0.5, max_tries=10):
#     """ rough sketch of top-two sampler """
#     first = sampler()
#     if random.random() < beta:
#         return first

#     number_of_tries = 0
#     while True:
#         number_of_tries += 1
#         second = sampler()
#         if number_of_tries > max_tries or accept_condition(first, second):
#             return second

