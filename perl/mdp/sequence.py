from .core import MDP

def is_unique(state):
    return len(set(state)) == len(state)

def is_increasing(state):
    return all(i < j for i, j in zip(state[:-1], state[1:]))

def Sequence(win_condition, k=5, n=None, discount=1, p_success=0.9):

    n = k if n is None else n
    # state is a list of numbers played so far
    # reward given iff at the end condition is satisfied

    def initial_states():
        return [(1, ())]

    def actions(state):
        return list(range(k))

    def next_state_reward(state, x):
        # concatenate tuple
        new = state + (x,)
        return (new, 0) if len(new) < n else (None, 1 if win_condition(new) else 0)

    def transitions(state, action):
        assert action in range(k)

        p_fail = 1 - p_success

        success = [(p_success, next_state_reward(state, action))]
        fails = [(p_fail / n, next_state_reward(state, random_action))
                for random_action in range(n)]
        return success + fails

    return MDP(initial_states, actions, transitions, discount)

