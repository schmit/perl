from .core import MDP
from ..distributions import Bernoulli

from collections import namedtuple, defaultdict
import random

def Chain(n=5, final_rew=10, exploit_rew=2, exit_prob=0.1):
    """
    Chain MDP (Strens, 2000; Poupart et al., 2006).
    The agent has two actions: Action 1 advances the agent along the chain,
    and Action 2 resets the agent to the first node. Action 1, when taken from
    the last node <n>, leaves the agent where it is and gives a reward of
    <final_rew>; all other rewards are 0.
    Action 2 always has a reward of <exploit_rew>.
    """

    def initial_states():
        return [(1, 1)]  # (prob, state)

    def actions(state):
        return [1, 2]

    def next_state_reward(state, action):
        if action == 1:
            if state < n:
                new_state = state + 1
                reward = 0
            else:
                new_state = n
                reward = final_rew
        else:
            new_state = 1
            reward = exploit_rew
        return new_state, reward

    def transitions(state, action):
        assert action in [1, 2]
        return [(1-exit_prob, next_state_reward(state, action)),
                (exit_prob, (None, 0))]

    return MDP(initial_states, actions, transitions, 1)



