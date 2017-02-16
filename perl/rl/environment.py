from collections import namedtuple

from ..mdp import find_all_states
from ..util import sample 

Environment = namedtuple("Environment", "states actions transition initial_states discount")

# for reference, an MDP is:
# MDP = namedtuple("MDP", "initial_states actions transitions discount")

def mdp_to_env(mdp):
    all_states = find_all_states(mdp)
    transition = lambda state, action: sample(mdp.transitions(state, action))

    return Environment(all_states, mdp.actions, transition, mdp.initial_states, mdp.discount)

def initial_value(env, values):
    value = 0
    for prob, state in env.initial_states():
        value += prob * values[state]
    return value

