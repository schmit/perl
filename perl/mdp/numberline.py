from .core import MDP
from ..distributions import Bernoulli


def Numberline(n=5, discount=1, p_success=0.6, p_die=0.1, p_left=0.5, p_right=0.5):
    assert 1 - p_success - p_die >= 0, "p_success + p_die > 1"

    def initial_states():
        return [(1, 0)]

    def actions(state):
        return [-1, 1]

    def next_state_reward(state, x):
        if state + x == n:
            return None, Bernoulli(p_right)
        if state + x == -n:
            return None, Bernoulli(p_left)
        return state + x, 0

    def transitions(state, action):
        assert action in [-1, 1]
        # what happens if action succeeds
        success = state + action
        success_state = success if abs(success) != n else None
        fail = state - action
        fail_state = fail if abs(fail) != n else None

        return [(p_success, next_state_reward(state, action)),
                (1-p_success-p_die, next_state_reward(state, -action)),
                (p_die, (None, 0))]

    return MDP(initial_states, actions, transitions, discount)

