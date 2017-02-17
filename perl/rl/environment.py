from collections import namedtuple

from ..mdp import find_all_states
from ..util import sample

Environment = namedtuple("Environment", "states actions transition initial_states discount")

# for reference, an MDP is:
# MDP = namedtuple("MDP", "initial_states actions transitions discount")

def mdp_to_env(mdp):
    """ Convert an MDP to an Environment """

    all_states = find_all_states(mdp)

    def transition(state, action):
        new_state, reward_distribution = sample(mdp.transitions(state, action))
        reward = sample_reward(reward_distribution)
        return new_state, reward

    return Environment(all_states, mdp.actions, transition, mdp.initial_states, mdp.discount)


def env_value(env, values):
    """
    Compute the overall value of the environmate
    based on probabilities over initial states

    Args:
        - env: Environment object
        - values: dictionary with state->value

    Returns:
        overall value of environment
    """
    return sum(prob * values[state] for prob, state in env.initial_states())

def sample_reward(reward):
    if isinstance(reward, int) or isinstance(reward, float):
        return reward
    return reward.sample()
