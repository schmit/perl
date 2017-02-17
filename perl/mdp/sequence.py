from collections import Counter
from .core import MDP
from ..distributions import Normal

def is_unique(state):
    """ Reward is 1 if all elements in state are unique """
    return 1 if len(set(state)) == len(state) else 0

def is_increasing(state):
    """ Reward is 1 if the sequence is increasing """
    return 1 if all(i < j for i, j in zip(state[:-1], state[1:])) else 0

def most_duplicates(state):
    """
    Reward is drawn from Normal distribution with
    mean the maximum number of duplicates.

    E.g. for state (1, 1, 2, 3) the reward is drawn from N(2, 1)
    as 1 occurs twice
    """
    return Normal(Counter(state).most_common(1)[0][1], 1)

def Sequence(win_condition, k=5, n=None, discount=1, p_success=0.9):

    n = n if n else k
    # state is a list of numbers played so far
    # reward given iff at the end all numbers are different

    def initial_states():
        return [(1, ())]

    def actions(state):
        return list(range(k))

    def next_state_reward(state, x):
        # concatenate tuple
        new = state + (x,)
        return (new, 0) if len(new) < n else (None, win_condition(new))

    def transitions(state, action):
        assert action in range(k)

        p_fail = 1 - p_success

        success = [(p_success, next_state_reward(state, action))]
        fails = [(p_fail / n, next_state_reward(state, random_action))
                for random_action in range(n)]
        return success + fails

    return MDP(initial_states, actions, transitions, discount)

