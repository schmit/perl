from collections import defaultdict
import math
import random
from .core import Algorithm
from ...mdp import MDP, policy_iteration, value_iteration
from ...rl.environment import mdp_to_env


class Rmax(Algorithm):
    """
    Rmax algorithm

    https://ie.technion.ac.il/~moshet/brafman02a.pdf
    """
    def __init__(self,
            mdp,
            discount=0.9,
            rmax_v=100,
            K=4):

        self.discount = discount
        self.R = defaultdict(lambda: 0)     # model for rewards
        self.P = defaultdict(list)          # model for transitions
        self.rmax_v = rmax_v                # upper bound on reward
        self.K = K                          # threshold for transition updates

        self.env = mdp_to_env(mdp)
        self.states = self.env.states
        self.actions = self.env.actions
        self.known = {}
        self.visits_sa = defaultdict(lambda: 0)
        self.reach_sa = defaultdict(set)
        self.trans_sa = defaultdict(lambda: defaultdict(lambda: 0))
        self.latest_values = None
        self.policy = {}

        if type(self.states[0]) in [type(0), type(0.0)]:
            self.s_heaven = min(self.states) - 1
        elif hasattr(self.states[0], '_fields'):
            self.s_heaven = tuple([random.random() for _ in self.states[0]._fields])
        else:
            self.s_heaven = tuple([random.random() for _ in range(len(self.states[0].__dict__))])

        for s in self.states:
            self.known[s] = 0
            self.policy[s] = random.choice(self.actions(s))
            for a in self.actions(s):
                self.R[(s,a)] = self.rmax_v
                self.P[(s,a)] = [(1, self.s_heaven)] # list of (prob, state)
        
        self.R[(self.s_heaven, 0)] = self.rmax_v
        self.P[(self.s_heaven, 0)] = [(1, self.s_heaven)]
        self.known[self.s_heaven] = 1

        # episode counter
        self.episode = 1

    def act(self, state):
        return self.policy[state]

    def learn(self, steps):
        for s, a, r, s2 in steps:
            self.visits_sa[(s,a)] += 1
            # update reward
            self.R[(s,a)] = (self.R[(s,a)] * (self.visits_sa[(s,a)] - 1) + r) / self.visits_sa[(s,a)]
            # update reached set
            if s2 is not None:
                self.reach_sa[(s,a)].add(s2)
                self.trans_sa[(s,a)][s2] += 1
                if len(self.reach_sa[(s,a)]) >= self.K:
                    # update the transition used for policy
                    self.known[s] = 1
                    self.P[(s,a)] = [(w/self.visits_sa[(s,a)], s3) for s3, w in self.trans_sa[(s,a)].items()]

        actions_t = {s4:self.actions(s4) for s4 in self.states}
        actions_t[self.s_heaven] = [0]

        mdp_t = create_mdp(self.P, self.R, actions_t, self.env.initial_states(), self.discount)

        value, policy = value_iteration(mdp_t, epsilon=1e-3, values=self.latest_values)
        self.latest_values = value
        self.policy = policy
        # need to complement policy for isolated states
        for s in self.states:
            if s not in self.policy:
                self.policy[s] = random.choice(self.actions(s))

        self.episode += 1

    @property
    def optimal_policy(self):
        # use all info about transitions
        opt_P = defaultdict(list)
        for s in self.states:
            for a in self.actions(s):
                opt_P[(s,a)] = [(w/self.visits_sa[(s,a)], s2) for s2, w in self.trans_sa[(s,a)].items()]
                if len(opt_P[(s,a)]) == 0:
                    # if we've never seen (s,a) yet
                    opt_P[(s,a)] = [(1.0/len(self.states), s2) for s2 in self.states]
        
        # replace rmax rewards with 0 for best guess
        opt_R = defaultdict(list)
        for (s,a), r in self.R.items():
            if r == self.rmax_v:
                r = 0
            if s in self.states:
                opt_R[(s,a)] = r

        mdp_opt = create_mdp(opt_P, opt_R, {s3:self.actions(s3) for s3 in self.states},
                             self.env.initial_states(), self.discount)
        value, policy = value_iteration(mdp_opt, epsilon=1e-3, values=None)
        self.latest_values = value
        return policy

    def __repr__(self):
        return "Rmax"

def create_mdp(trans_v, reward_v, action_v, initial, discount):
    """
        Returns an MDP given by:
        - trans_v: transitions, dict: (s,a) -> list of (prob, state)
        - reward_v: mean reward, dict: (s,a) -> r.
        - action_v: dict: s -> set of actions.
        - initial: initial states, list of (prob, state).
        - discount: discount factor.
    """

    def initial_states():
        return initial

    def actions(state):
        return action_v[state]

    def transitions(state, action):
        r = reward_v[(state, action)]
        return [(prob, (s, r)) for prob, s in trans_v[(state, action)]]

    return MDP(initial_states, actions, transitions, discount)


def display_mdp(mdp, states):
    for s in states:
        print("=====================")
        print("State: {}.".format(s))
        print("Actions: {}.".format(mdp.actions(s)))
        print("Transitions: ")
        for a in mdp.actions(s):
            print(a, mdp.transitions(s, a))





