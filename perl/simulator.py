from perl.util import sample, sars

def run(mdp, policy, verbose=False):
    state, idx = sample(mdp.initial_states())
    history = [state]

    if verbose:
        print("Initial state: {}".format(state))

    while state is not None:
        action = policy[state]
        (state, reward), idx = sample(mdp.transitions(state, action))
        history += [action, reward, state]
        if verbose:
            print("Action: {}\tReward: {}\tNew state: {}".format(action, reward, state))

    return history

def discounted_reward(history, discount):
    _, _, rewards, _ = zip(*sars(history))
    return sum(discount**i * reward for i, reward in enumerate(rewards))
