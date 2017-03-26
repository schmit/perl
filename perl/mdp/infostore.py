from .core import MDP
from ..distributions import Normal

from collections import namedtuple, defaultdict

def InfoStore(mdp, c, sigma2):

    # states = (info, state)
    # info = -1, 0, 1 (first state, world, infostore resp.)

    def initial_states():
        return [(1, (-1, 0))]

    def actions(state):
        info, s = state
        if info == -1:
            return [0, 1]
        else:
            return mdp.actions(s)

    def transitions(state, action):
        info, s = state

        if info == -1:
            if action == 1:
                # enter info store
                return [(prob, ((1, s1), -c)) for prob, s1 in mdp.initial_states()]
            else:
                # enter real world
                return [(prob, ((0, s1), 0)) for prob, s1 in mdp.initial_states()]
        else:
            transitions = []
            for prob, pair in mdp.transitions(s, action):
                next_state, reward = pair
                if info == 0 and type(reward) == type(Normal(0, 1)):
                    reward = Normal(reward.mu, reward.sigma + sigma2)
                if next_state is None:
                    transitions.append((prob, (None, reward)))
                else:
                    transitions.append((prob, ((info, next_state), reward)))
        return transitions

    return MDP(initial_states, actions, transitions, mdp.discount)
