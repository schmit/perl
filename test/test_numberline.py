from perl.mdp.numberline import Numberline


numberline = Numberline()


def test_actions():
    assert numberline.actions(0) == [-1, 1]

def test_transitions():
    assert numberline.transitions(0, 1) == [(0.6, 1, 0), (0.4, 0, 0)]
    assert numberline.transitions(4, 1) == [(0.6, None, 1), (0.4, 4, 0)]
    assert numberline.transitions(-4, -1) == [(0.6, None, 0), (0.4, -4, 0)]
