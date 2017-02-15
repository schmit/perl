from perl.util import sample, sars

def run(env, policy, verbose=False):
    state, idx = sample(env.initial_states())
    history = [state]

    if verbose:
        print("Initial state: {}".format(state))

    while state is not None:
        action = policy[state]
        (state, reward), idx = env.transition(state, action)
        history += [action, reward, state]
        if verbose:
            print("Action: {}\tReward: {}\tNew state: {}".format(action, reward, state))

    return history

def discounted_reward(history, discount):
    _, _, rewards, _ = zip(*sars(history))
    return sum(discount**i * reward for i, reward in enumerate(rewards))
