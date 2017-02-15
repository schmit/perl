from ..util import sample, sars

def episode(env, algo, verbose=False):
    algo.init_episode()

    state, _ = sample(env.initial_states())
    history = [state]

    if verbose:
        print("Initial state: {}".format(state))

    while state is not None:
        action = algo.act(state)
        (state, reward), idx = env.transition(state, action)
        history += [action, reward, state]
        if verbose:
            print("Action: {}\tReward: {}\tNew state: {}".format(action, reward, state))

    return history

def live(env, algo, num_episodes=1, verbose=None):
    rewards = []
    total_rewards = 0
    for epi in range(num_episodes):
        # run an episode
        history = episode(env, algo)
        steps = sars(history)
        algo.learn(steps)

        # log performance
        discounted_reward = sum(env.discount**t * r
                for t, (_, _, r, _) in enumerate(steps))
        rewards.append(discounted_reward)
        total_rewards += discounted_reward

        # print progress
        if verbose and (epi+1) % verbose == 0:
            print("ep: {} - d.reward: {:.3f}".format(epi+1, total_rewards))

    return rewards


def discounted_reward(history, discount):
    _, _, rewards, _ = zip(*sars(history))
    return sum(discount**i * reward for i, reward in enumerate(rewards))

