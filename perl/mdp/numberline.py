from .base import MDP


def Numberline(n=5, discount=0.9):
    def initial_state():
        return [(1, 0)]

    def actions(state):
        return [-1, 1]

    def transitions(state, action):
        assert action in [-1, 1]
        # what happens if action succeeds
        success = state + action
        success_reward = 1 if success == n else 0
        success_state = success if abs(success) != n else None

        return [(0.6, success_state, success_reward),
                (0.4, state, 0)]

    return MDP(initial_state, actions, transitions, discount)

