import random

def sample(options):
    """
    Randomly sample an element from
    [(probability, element)]

    Returns both the random element and its index
    """
    i, x = 0, random.random()
    for prob, elem in options:
        if x < prob:
            return elem, i
        x -= prob
        i += 1
    raise IndexError

def sars(history):
    """
    Extract (state, action, reward, new_state) tuples from history
    """
    steps = int((len(history)-1)/3)
    return [history[(3*i):(3*i+4)] for i in range(steps)]



