from .core import MDP


def Numberline(n=5, discount=1, p_success=0.6, p_die=0.1):
    assert 1 - p_success - p_die >= 0, "p_success + p_die > 1"

    def initial_states():
        return [(1, 0)]

    def actions(state):
        return [-1, 1]

    def transitions(state, action):
        assert action in [-1, 1]
        # what happens if action succeeds
        success = state + action
        success_reward = 1 if success == n else 0
        success_state = success if abs(success) != n else None

        return [(p_success, (success_state, success_reward)),
                (1-p_success-p_die, (state, 0)),
                (p_die, (None, 0))]

    return MDP(initial_states, actions, transitions, discount)

