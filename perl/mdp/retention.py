from .core import MDP

def Retention(n, m, p_c=0.1, p_die=0.05, discount=1):
    """
    Retention MDP.
    n = number of types of users. capacity in (1,...,n)
    m = number of actions / doses

    state (c,d) = (capacity of user, total dose given so far)
    if d > c => die, else reward d.
    with small probability c increases over time.
    """
    def initial_states():
        # all types of users are equally likely
        return [(1/n, (i, 0)) for i in range(1,n+1)]

    def actions(state):
        # any dose {0, ..., m}
        return list(range(m+1))

    def from_state(state, action):
        c, d = state
        return (c, d+action)

    def transitions(state, action):
        assert action in list(range(m+1))

        # reward equals 

        trans = []

        success_s = from_state(state, action)
        if success_s[1] <= success_s[0]:
            # the user stays
            success_r = success_s[1]
            trans.append((1-p_die-p_c, (success_s, success_r)))
            # the user may get more resistance
            success_up = (min(success_s[0]+1, n), success_s[1])
            trans.append((p_c, (success_up, success_r)))
        else:
            # overdose and death
            success_r = -success_s[0]
            trans.append((1-p_die, (None, success_r)))
        # organic abandon
        trans.append((p_die, (None, 0)))

        return trans

    return MDP(initial_states, actions, transitions, discount)

