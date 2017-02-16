from collections import defaultdict
import math
import random
from .core import Algorithm


def greedy(qvals, *args):
    # add a bit of randomness to break ties
    action, qval = max(qvals, key=lambda t: t[1] + random.random() * 0.001)
    return action

def epsgreedy(qvals, num_episode):
    epsilon = 1/100 + 1/math.sqrt(num_episode + 1)
    if random.random() < epsilon:
        action, _ = random.choice(qvals)
    else:
        action = greedy(qvals)
    return action



class Qlearning(Algorithm):
    def __init__(self, env, lr=0.1, discount=0.9, dithering=epsgreedy):
        self.lr = lr
        self.discount = discount
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))
        self.dithering = dithering
        self.actions = env.actions

        # episode counter
        self.episode = 1

    def act(self, state):
        qvals = [(action, self.Q[state][action]) for action in self.actions(state)]
        return self.dithering(qvals, self.episode)

    def learn(self, steps):
        for state, action, reward, new_state in steps:
            residual = reward - self.Q[state][action]
            if new_state is not None:
                residual += self.discount * max(qval for action, qval in self.Q[new_state].items())

            self.Q[state][action] += self.lr * residual

        self.episode += 1


