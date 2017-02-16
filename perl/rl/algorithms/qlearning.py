from collections import defaultdict
import math
import random
from .core import Algorithm


def greedy(qvals, *args):
    """ Take action with highest Q-value """
    # add a bit of randomness to break ties
    action, qval = max(qvals, key=lambda t: t[1] + random.random() * 0.001)
    return action

def epsgreedy(qvals, num_episode):
    """
    With probability 1/100 + 1/sqrt(num_episode + 1)
    take a random action, otherwise take the greedy action
    """
    epsilon = 1/100 + 1/math.sqrt(num_episode + 1)
    if random.random() < epsilon:
        action, _ = random.choice(qvals)
    else:
        action = greedy(qvals)
    return action


class Qlearning(Algorithm):
    """
    Q-learning algorithm

    Currently learns using TD(0), using other TD learning methods is todo
    """
    def __init__(self,
            env,
            lr=lambda num_episodes: min(1, 1/num_episodes**0.5),
            discount=0.9,
            dithering=epsgreedy):
        self.lr = lr
        self.discount = discount
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))
        self.dithering = dithering

        self.states = env.states
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

            self.Q[state][action] += self.lr(self.episode) * residual

        self.episode += 1

    @property
    def optimal_policy(self):
        policy = {state: max((action for action in self.actions(state)),
                             key=lambda action: self.Q[state][action])
                    for state in self.states}
        return policy

    def __repr__(self):
        return "Q-learning"

