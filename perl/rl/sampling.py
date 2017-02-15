import random

from ..bayesian import Normal, Dirichlet
from ..mdp import MDP, find_all_states, value_iteration
from ..simulator import run
from ..util import sars

def create_sampler(mdp, d_reward = lambda: Normal(0, 1, 1)):
    # sorted so we can always recover the order from the indexer
    all_states = sorted(list(find_all_states(mdp)))
    nstates = len(all_states)

    posterior_transitions = {(state, action): Dirichlet(nstates + 1, 1/(nstates+2))
            for state in all_states for action in mdp.actions(state)}

    posterior_rewards = {(state, action): d_reward()
                 for state in all_states for action in mdp.actions(state)}

    sampler = lambda: sample_mdp(mdp,
                                 all_states,
                                 posterior_transitions,
                                 posterior_rewards)

    state_indexer = {state: idx for idx, state in enumerate(all_states)}
    state_indexer[None] = nstates

    return sampler, posterior_transitions, posterior_rewards, state_indexer

def sample_mdp(mdp, all_states, posterior_transitions, posterior_rewards):
    sampled_transitions = {(state, action): posterior.sample()
            for (state, action), posterior in posterior_transitions.items()}
    sampled_rewards = {(state, action): posterior.sample()
            for (state, action), posterior in posterior_rewards.items()}

    def transitions(state, action):
        # note do not consider the last sampled transition probability
        # as that is the probability to end the MDP
        return [(p, (all_states[i], sampled_rewards[(state, action)]))
                for i, p in enumerate(sampled_transitions[(state, action)][:-1])]

    # make sure discount is less than one because sampled MDP might have
    # infinite cycles
    return MDP(mdp.initial_states, mdp.actions, transitions, min(0.9, mdp.discount))

def map_mdp(mdp, all_states, posterior_transitions, posterior_rewards):
    map_transitions = {(state, action): posterior.map()
            for (state, action), posterior in posterior_transitions.items()}
    map_rewards = {(state, action): posterior.map()
            for (state, action), posterior in posterior_rewards.items()}

    def transitions(state, action):
        # note do not consider the last sampled transition probability
        # as that is the probability to end the MDP
        return [(p, (all_states[i], map_rewards[(state, action)]))
                for i, p in enumerate(map_transitions[(state, action)][:-1])]

    return MDP(mdp.initial_states, mdp.actions, transitions, mdp.discount)

def update_posteriors(steps, posterior_transitions, posterior_rewards, state_indexer):
    for state, action, reward, new_state in steps:
        # update transition:
        posterior_transitions[(state, action)].update(state_indexer[new_state])
        # update reward
        posterior_rewards[(state, action)].update(reward)

def posterior_sampling(mdp, episodes, d_reward=lambda: Normal(0, 1, 1), verbose=None):
    sampler, posterior_transitions, posterior_rewards, indexer = create_sampler(mdp, d_reward)
    episode_rewards = []

    # initial values are none
    values = None

    cum_rewards = 0
    for episode in range(episodes):
        # start with low accuracy value iteration for efficiency
        # also, after some samples reuse old value for faster value_iteration
        values, policy = value_iteration(sampler(),
                epsilon=10/(episode+1)**1.8,
                values=None if episode < 20 else values)

        steps = sars(run(mdp, policy))
        update_posteriors(steps, posterior_transitions, posterior_rewards, indexer)

        # log rewards
        _, _, rewards, _ = zip(*steps)
        total_reward = sum(mdp.discount**t * reward
                for t, reward in enumerate(rewards))
        cum_rewards += total_reward
        episode_rewards.append(total_reward)

        if verbose and episode % verbose == 0:
            print("Episode: {}, cumulative reward: {}".format(episode+1, cum_rewards))


    return sampler, posterior_transitions, posterior_rewards, indexer, episode_rewards


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

