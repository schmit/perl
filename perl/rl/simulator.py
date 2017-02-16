import math
from statistics import mean, stdev
import time

from .algorithms import FixedPolicy
from ..util import sample, sars

def episode(env, algo, verbose=False):
    """
    Runs a single episode of a learning algorithm in an environment

    Args:
        - env: Environment
        - algo: Learning algorithm (e.g. Qlearning)
        - verbose: Bool indicating whether steps should be printed

    Returns:
        history list: [s, a, r, s, a, r, s]
    """
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
    """
    Run a learning algorithm on an environment

    Args:
        - env: Learning Environment
        - algo: Learning algorithm (e.g. Qlearning)
        - num_episodes: Number of episodes in the environment
        - verbose: Integer specifying how often progress should be printed (default: 0)

    Returns:
        list of discounted sum of rewards for each episode
    """
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

def reward_path(env, algo, num_episodes, num_repeats=20,
        num_test_episodes=1000, verbose=True):
    """
    Run <live> for <num_repeats> times for <num_episodes> length,
    and approximate the value of the current policy between calls to <live>.

    Args:
        - env: Environment
        - algo: Learning algorithm
        - num_episodes: Number of episodes for each <live> call
        - num_repeats: Number of calls to <live>
        - num_test_episodes: Number of episodes to test the current best policy
        - verbose: Bool indicating output

    Returns:
        List with performance metrics after each repeat of <live>
        [(num_episodes, mean_learning_reward, sd_learning_reward,
          mean_test_reward, sd_test_reward)]
    """
    path = []
    total_episodes = 0

    t_start = time.time()
    for repeat in range(num_repeats):
        total_episodes += num_episodes
        learning_rewards = live(env, algo, num_episodes)

        current_policy = algo.optimal_policy
        testing_rewards = live(env, FixedPolicy(current_policy), num_test_episodes)

        path.append((total_episodes,
                     mean(learning_rewards),
                     stdev(learning_rewards) / math.sqrt(num_episodes),
                     mean(testing_rewards),
                     stdev(testing_rewards) / math.sqrt(num_test_episodes)))

        if verbose:
            print("{}/{} done...".format(total_episodes, num_episodes * num_repeats))

    t_end = time.time()
    print("[{}] Ran path in {} seconds.".format(algo, t_end - t_start))

    return path


def discounted_reward(history, discount):
    _, _, rewards, _ = zip(*sars(history))
    return sum(discount**i * reward for i, reward in enumerate(rewards))


