from .core import MDP

from collections import namedtuple, defaultdict
import random
from scipy.stats import binom

def StockPreComp(M, D, P, S=100, T=10, price_set=[1,2,3,4,5]):
    """
    Dynamic Pricing MDP:
    M = market transition matrix.
    D = market_state x num_incoming_cust probability matrix.
    P = market_state x prices = prob of buying in the market state at price p
    S = initial stock.
    T = total number of days.
    """

    def initial_states():
        # state = (market state, day, remaining stock)
        return [(1, (0, 0, S))]

    def actions(state):
        return range(len(price_set))

    def precompute_transitions():

        transition = {}

        for m_state in range(M.shape[0]):
            for day in range(T+1):
                for stock in range(S+1):
                    for action in range(len(price_set)):
                        state = m_state, day, stock
                        trans = []
                        num_m, num_c = D.shape
                        p = P[m_state, action] # prob of a new customer buying at that price
                        for c_i in range(num_c):
                            p_ci = D[m_state, c_i]   # prob of c_i customers
                            for w_i in range(c_i+1):
                                p_wi = binom.pmf(w_i, c_i, p) # prob of w_i purchases out of c_i
                                rew = min(w_i, stock) * price_set[action]
                                for new_market in range(num_m):
                                    p_nm = M[m_state, new_market]
                                    if day == T - 1:
                                        trans.append((p_nm * p_ci * p_wi, (None, rew)))
                                    else:
                                        trans.append((p_nm * p_ci * p_wi, ((new_market, day+1, stock-min(w_i, stock)), rew)))

                        transition[tuple((state, action))] = trans

        return transition

    memory = precompute_transitions()

    def transitions(state, action):
        return memory[tuple((state, action))]

    return MDP(initial_states, actions, transitions, 1)



