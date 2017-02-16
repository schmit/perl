from collections import namedtuple

from ..mdp import find_all_states
from ..util import sample

Environment = namedtuple("Environment", "states actions transition initial_states discount")

# for reference, an MDP is:
# MDP = namedtuple("MDP", "initial_states actions transitions discount")

def mdp_to_env(mdp):
    """ Convert an MDP to an Environment """
    all_states = find_all_states(mdp)
    transition = lambda state, action: sample(mdp.transitions(state, action))

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
    return sum(prob * value for prob, val in env.initial_states())
