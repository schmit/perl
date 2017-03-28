from .core import MDP
from ..distributions import Normal

from collections import namedtuple, defaultdict
import random

State = namedtuple("State", "depth location")

def NormalTriangle(max_depth=5, means=[1, 1.25, 1.5, 1.75, 2], sigma2=[1]):
    """
    Normal Triangle MDP:
    Triangle graph with diamond pattern where on each action
    you descend 1 level and either go left or right.
    Each transition has a Normal(m, sigma2) reward where m is drawn from probs.
    """
    reward_means = defaultdict(lambda: random.choice(means))
    reward_stds = defaultdict(lambda: random.choice(sigma2))

    def initial_states():
        return [(1, State(0, 0))]

    def actions(state):
        return [-1, 1]

    def next_state_reward(state, action):
        depth = state.depth + 1
        location = state.location + action
        new_state = State(depth, location) if depth < max_depth else None
        return new_state, Normal(reward_means[(state, action)], reward_stds[(state, action)])

    def transitions(state, action):
        assert action in [-1, 1]
        return [(1, next_state_reward(state, action))]

    return MDP(initial_states, actions, transitions, 1)



