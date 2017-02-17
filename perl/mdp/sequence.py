from .core import MDP

def is_unique(state):
    return len(set(state)) == len(state)


def Sequence(win_condition, n=5, discount=1, p_success=0.9):

    # state is a list of numbers played so far
    # reward given iff at the end all numbers are different

    def initial_states():
        return [(1, ())]

    def actions(state):
        return list(range(n))

    def next_state_reward(state, x):
        # concatenate tuple
        new = state + (x,)
        return (new, 0) if len(new) < n else (None, 1 if win_condition(new) else 0)

    def transitions(state, action):
        assert action in range(n)

        p_fail = 1 - p_success

        success = [(p_success, next_state_reward(state, action))]
        fails = [(p_fail / n, next_state_reward(state, random_action))
                for random_action in range(n)]
        return success + fails

    return MDP(initial_states, actions, transitions, discount)

