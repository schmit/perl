from perl.distributions import *

n = Normal(0, 1)
b = Bernoulli(0.4)
r = Rademacher()

def test_mean():
    assert n.mean == 0
    assert b.mean == 0.4
    assert r.mean == 0

def test_sample():
    assert b.sample() in [0, 1]
    assert r.sample() in [-1, 1]


