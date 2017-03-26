from .core import MDP

import random

def f_sum(x, y):
    return x + y + random.gauss(0, min(x,y)/7)

def f_prod(x, y):
    return x * y + random.gauss(0, min(x,y)/7)

def Optimizer(f, xmin=0, xmax=5, ymin=0, ymax=0, p_random=0.1, p_die=0.05, discount=1):
    """
    Optimizer MDP.
    We try to optimize reward given by function f(x,y) [can be random].
    The valid states are the integer lattice with
    xmin <= x <= xmax and ymin <= y <= ymax.
    There are 5 actions, move up, down, left, right or stay.
    It happens with prob 1-p_random-p_die.

    With probability p_die, we die.
    With prob p_random we take a random action.
    """
    def initial_states():
        return [(1, (int((xmax+xmin)/2), int((ymax+ymin)/2)))]

    def actions(state):
        # 0 -> x -= 1, 1 -> x += 1
        # 2 -> y -= 1, 3 -> y += 1
        # 4 -> stay
        return [0, 1, 2, 3, 4]

    def from_state(state, a):
        x, y = state
        if a == 0:
            return (max(x-1, xmin), y)
        if a == 1:
            return (min(x+1, xmax), y)
        if a == 2:
            return (x, max(y-1, ymin))
        if a == 3:
            return (x, min(y+1, ymax))
        return (x, y)

    def transitions(state, action):
        assert action in [0, 1, 2, 3, 4]
        # what happens if action succeeds
        success_state = from_state(state, action)
        success_reward = f(success_state[0], success_state[1])

        n = len(actions(state))
        random_s = [from_state(state, a) for a in actions(state)]
        random_t = [(p_random/n, (s, f(s[0], s[1]))) for s in random_s]

        random_t.append((1-p_random-p_die, (success_state, success_reward)))
        random_t.append((p_die, (None, 0)))

        return random_t

    return MDP(initial_states, actions, transitions, discount)

