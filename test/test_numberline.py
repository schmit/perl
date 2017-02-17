from perl.mdp.numberline import Numberline


numberline = Numberline()


def test_actions():
    assert numberline.actions(0) == [-1, 1]

