from .core import MDP

def Norepeat(n=5, discount=1, p_success=0.9):

    # state is a list of numbers played so far
    # reward given iff at the end all numbers are different

    def initial_states():
        return [(1, (0))]

    def actions(state):
        return [i for i in range(1, n+1)]

    def transitions(state, action):
        assert action in range(1, n+1)
        # what happens if action succeeds

        if len(state) == n+1:
            # after collecting n numbers, game is over
            return [(1, (None, 0))]

        next_state = tuple(state[:] + [action])
        next_reward = 1 if set(next_state) == set(range(n+1)) else 0
        trans = [(p_success, (next_state, next_reward))]
        eps = (1-p_success)/(n-1)

        for num in range(1, n+1):
            if num == action:
                continue
            num_state = tuple(state[:] + [num])
            num_reward = 1 if set(num_state) == set(range(n+1)) else 0
            trans.append((eps, (num_state, num_reward)))

        return trans

    return MDP(initial_states, actions, transitions, discount)
