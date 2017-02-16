from collections import defaultdict
import random
from .base import Algorithm
from ...mdp import bellman




class Qlearning(Algorithm):
    def __init__(self, env, lr=0.001, exploration=greedy):
        self.lr = lr
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))
        self.exploration = exploration

    def act(self, state):
        qvals = [(action, self.Q[state][action]) for action in self.Q[state].keys()]
        return self.exploration(qvals, param)

    def learn(self, steps):
        pass


def greedy(qvals):
    action, qval = max(qvals, key=lambda t: t[1])
    return action

def epsgreedy(qvals, epsilon=0.1):
    if random.random() < epsilon:
        action, qval = random.choice(qvals)
    else:
        action, qval = greedy(qvals)
    return action



