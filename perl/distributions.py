from collections import namedtuple

import random


class Normal:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        return random.gauss(self.mu, self.sigma)

    @property
    def mean(self):
        return self.mu

    def __repr__(self):
        return "Normal({:.2f}, {:.2f})".format(self.mu, self.sigma)


class Bernoulli:
    def __init__(self, p):
        self.p = p

    def sample(self):
        return 1 if random.random() < self.p else 0

    @property
    def mean(self):
        return self.p

    def __repr__(self):
        return "Bernoulli({:.2f})".format(self.p)


class Rademacher:
    def sample(self):
        return 1 if random.random() < 0.5 else -1

    @property
    def mean(self):
        return 0

    def __repr__(self):
        return "Rademacher"


