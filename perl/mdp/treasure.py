from collections import namedtuple
from .core import MDP

State = namedtuple("State", "x y captured")


def Treasure(width, height, treasures, p_die=0.01, discount=1):
    """
    Treasures have been hidden underground in a grid.
    Agent starts randomly within grid.

    Args:
        width: width of grid
        height: height of grid
        treasures: list/set of locations of treasures
        p_die: probability of dying
        discount: discount factor

    Returns:
        MDP
    """
    size = width * height

    def initial_states():
        return [(1/size, State(x, y, frozenset()))
                for x in range(width) for y in range(height)]

    def actions(state):
        A = ["dig"]
        if state.x > 0:
            A.append("left")
        if state.x < width-1:
            A.append("right")
        if state.y > 0:
            A.append("up")
        if state.y < height-1:
            A.append("down")
        return A

    def transitions(state, action):
        death = (p_die, (None, -1))

        if action == "dig":
            if (state.x, state.y) in treasures and (state.x, state.y) not in state.captured:
                # have not captured the chest yet
                return [death,
                        (1-p_die,
                            (State(state.x, state.y,
                                   state.captured.union([(state.x, state.y)])),
                             1))]
            else:
                return [death, (1-p_die, (state, 0))]

        if action == "left":
            return [death,
                    (1-p_die, (State(state.x-1, state.y, frozenset(state.captured)), 0))]

        if action == "right":
            return [death,
                    (1-p_die, (State(state.x+1, state.y, frozenset(state.captured)), 0))]

        if action == "up":
            return [death,
                    (1-p_die, (State(state.x, state.y-1, frozenset(state.captured)), 0))]

        if action == "down":
            return [death,
                    (1-p_die, (State(state.x, state.y+1, frozenset(state.captured)), 0))]

    return MDP(initial_states, actions, transitions, discount)
