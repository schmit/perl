from .core import MDP
from ..distributions import Bernoulli

from collections import namedtuple, defaultdict
import random

State = namedtuple("State", "depth location")

def Triangle(max_depth=5, probs=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Triangle MDP:

    Triangle graph with diamond pattern where on each action
    you descend 1 level and either go left or right.
    Each transition has a Bernoulli(p) reward where p is drawn from probs.


    """
    reward_distributions = defaultdict(lambda: random.choice(probs))

    def initial_states():
        return [(1, State(0, 0))]

    def actions(state):
        return [-1, 1]

    def next_state_reward(state, action):
        depth = state.depth + 1
        location = state.location + action
        new_state = State(depth, location) if depth < max_depth else None
        return new_state, Bernoulli(reward_distributions[(state, action)])


    def transitions(state, action):
        assert action in [-1, 1]
        return [(1, next_state_reward(state, action))]

    return MDP(initial_states, actions, transitions, 1)



