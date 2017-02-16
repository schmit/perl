from collections import defaultdict
import random
from .base import Algorithm
from ...mdp import bellman




class Qlearning(Algorithm):
    def __init__(self, env, lr=0.001, discount=0.9, dithering=greedy):
        self.lr = lr
        self.discount = discount
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))
        self.dithering = dithering

    def act(self, state):
        qvals = [(action, self.Q[state][action]) for action in self.Q[state].keys()]
        return self.dithering(qvals)

    def learn(self, steps):
        for state, action, reward, new_state in steps:
            residual = reward - self.Q[state][action]
            if new_state is not None:
                residual += self.discount * max(action for action, qval in self.Q[new_state].items())

            self.Q[state][action] += self.lr * residual


def greedy(qvals):
    action, qval = max(qvals, key=lambda t: t[1])
    return action

def epsgreedy(qvals, epsilon=0.1):
    if random.random() < epsilon:
        action, qval = random.choice(qvals)
    else:
        action, qval = greedy(qvals)
    return action



