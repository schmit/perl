
from perl.mdp import MDP, value_iteration
from perl.mdp import find_all_states
from perl.mdp.numberline import Numberline

numberline = Numberline()

def test_find_all_states():
    all_states = find_all_states(numberline)
    assert list(range(-4, 5)) == all_states



