from statistics import mean, stdev
import time

from .algorithms import FixedPolicy
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

def reward_path(env, name, algo, num_episodes, num_repeats=20,
        num_test_episodes=1000, verbose=True):
    path = []
    total_episodes = 0

    t_start = time.time()
    for repeat in range(num_repeats):
        total_episodes += num_episodes
        learning_rewards = live(env, algo, num_episodes)

        current_policy = algo.optimal_policy
        testing_rewards = live(env, FixedPolicy(current_policy), num_test_episodes)

        path.append((total_episodes, mean(learning_rewards), stdev(learning_rewards),
                     mean(testing_rewards), stdev(testing_rewards)))

        if verbose:
            print("{}/{} done...".format(total_episodes, num_episodes * num_repeats))

    t_end = time.time()
    print("[{}] Ran path in {} seconds.".format(name, t_end - t_start))

    return path


def discounted_reward(history, discount):
    _, _, rewards, _ = zip(*sars(history))
    return sum(discount**i * reward for i, reward in enumerate(rewards))


