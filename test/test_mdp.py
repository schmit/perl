
from perl.mdp import MDP, value_iteration
from perl.mdp.base import find_all_states
from perl.mdp.numberline import Numberline

numberline = Numberline()

def test_find_all_states():
    all_states = find_all_states(numberline)
    assert set(range(-4, 5)) == all_states
