import random
import time

def eta(start_time, percentage_done):
    time_elapsed = time.time() - start_time
    return (1-percentage_done)/percentage_done * time_elapsed

def sample(options):
    """
    Randomly sample an element from
    [(probability, element)]

    Returns both the random element and its index
    """
    x = random.random()
    for prob, elem in options:
        if x < prob:
            return elem
        x -= prob
    raise IndexError

def sars(history):
    """
    Extract (state, action, reward, new_state) tuples from history
    """
    steps = int((len(history)-1)/3)
    return [history[(3*i):(3*i+4)] for i in range(steps)]

def repeat(fn, times, verbose=False):
    start_time = time.time()

    for t in range(times):
        if verbose and (t+1) % verbose == 0:
            percentage_done = max(0.0001, t / times)
            print("{:3.2f}% done... eta: {:4.1f} seconds".format(
                100*percentage_done,
                eta(start_time, percentage_done)))

        yield fn()



