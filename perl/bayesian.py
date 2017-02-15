"""
Holds conjugate prior distributions that can be updated with new data
"""

import numpy as np
import math
import random



class Normal:
    def __init__(self, mu0, sigma0, sigma):
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.sigma = sigma

    def update(self, X):
        gamma = self.sigma0**2 / (self.sigma**2 + self.sigma0**2)
        self.mu0 = gamma * X + (1-gamma) * self.mu0
        self.sigma0 = math.sqrt(self.sigma**2 * self.sigma0**2 /
                        (self.sigma**2 + self.sigma0**2))

    def sample(self):
        return random.gauss(self.mu0, self.sigma0)

    @property
    def map(self):
        return self.mu0

    def __repr__(self):
        return "Normal({:.2f},{:.2f})".format(self.mu0, self.sigma0)


class Dirichlet:
    def __init__(self, k, a0):
        self.params = [a0 for _ in range(k)]

    def update(self, X):
        self.params[X] += 1

    def sample(self):
        return list(np.random.dirichlet(self.params))

    @property
    def map(self):
        total = sum(self.params)
        return [p/total for p in self.params]

    def __repr__(self):
        return "Dirichlet({})".format(self.params)


class Beta(Dirichlet):
    def __init__(self, alpha, beta):
        self.params = [beta, alpha]

    def sample(self):
        return np.random.dirichlet(self.params)[1]

    @property
    def map(self):
        return self.params[1] / sum(self.params)

    def __repr__(self):
        return "Beta({}, {})".format(self.params[1], self.params[0])

