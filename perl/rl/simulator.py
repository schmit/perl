import math
from statistics import mean, stdev
import time
from collections import defaultdict

from .algorithms import FixedPolicy
from ..mdp import policy_iteration, value_iteration
from .environment import env_value, mdp_to_env
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

    state = sample(env.initial_states())
    history = [state]

    if verbose:
        print("Initial state: {}".format(state))

    while state is not None:
        action = algo.act(state)
        state, reward = env.transition(state, action)
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

def reward_path(env, algo, num_episodes, log_every=20,
        num_test_episodes=1000, mdp=None, verbose=True):
    """
    Run <live> for <num_episodes> logging data every <log_every> episodes,
    and approximate the value of the current policy between calls to <live>.

    Args:
        - env: Environment
        - algo: Learning algorithm
        - num_episodes: Total number of episodes
        - log_every: Number of calls for each <live> call
        - num_test_episodes: Number of episodes to test the current best policy
        - mdp (optional): if MDP is supplied, uses policy iteration to compute exact performance
        - verbose: Bool indicating output

    Returns:
        List with performance metrics after each repeat of <live>
        [(num_episodes, mean_learning_reward, sd_learning_reward,
          mean_test_reward, sd_test_reward)]
    """
    path = []
    total_episodes = 0
    num_rounds = int(num_episodes / log_every)

    t_start = time.time()
    for repeat in range(num_rounds):
        # learn
        total_episodes += log_every
        learning_rewards = live(env, algo, log_every)

        # test
        current_policy = algo.optimal_policy
        if mdp is None:
            testing_rewards = live(env, FixedPolicy(current_policy), num_test_episodes)
            performance = (total_episodes,
                        mean(learning_rewards),
                        stdev(learning_rewards) / math.sqrt(log_every),
                        mean(testing_rewards),
                        stdev(testing_rewards) / math.sqrt(num_test_episodes))
            path.append(performance)
        else:
            testing_rewards = env_value(env, policy_iteration(mdp, current_policy))
            performance = (total_episodes,
                        mean(learning_rewards),
                        stdev(learning_rewards) / math.sqrt(log_every),
                        testing_rewards)
            path.append(performance)

        # print
        if verbose:
            print("{:5d}/{:5d} done... ({:2.2f} | {:2.2f})".format(total_episodes,
                num_episodes,
                performance[1],
                performance[3]))

    t_end = time.time()
    if verbose:
        print("[{}] simulation took {} seconds.".format(algo, t_end - t_start))

    return path

def discounted_reward(history, discount):
    _, _, rewards, _ = zip(*sars(history))
    return sum(discount**i * reward for i, reward in enumerate(rewards))


def comparison_sim(mdp, algo_list, algo_names, algo_params,
                   num_sims=20, num_episodes=100, log_every=5, verbose=0):

    t1 = time.time()

    results, times = defaultdict(list), defaultdict(list)
    env = mdp_to_env(mdp)

    if verbose:
        opt_val, opt_pol = value_iteration(mdp)
        max_val = env_value(env, opt_val)
        print("Max Value of the MDP is {}.".format(max_val))

    for sim in range(num_sims):
        if verbose:
            # print("=========================================================")
            print("{} | Starting sim {}/{} after {} seconds.".format(verbose, sim, num_sims, time.time()-t1))
            # print("=========================================================")

        for i in range(len(algo_list)):
            algo = algo_list[i](**algo_params[i])
            algo.name = algo_names[i]
            t_start = time.time()
            path = reward_path(env, algo, num_episodes, log_every, mdp=mdp, verbose=False)
            results[algo.name].append(path)
            times[algo.name].append(time.time()-t_start)
            # if verbose:
            #     print("=====================================")
            #     print("{} | Sim {} | Took {} s.".format(algo.name, sim, times[algo.name][-1]))
            #     print("mean performance of best policy (episode, performance):")
            #     print([(elt[0], elt[3]) for elt in path])

    t2 = time.time()
    print("=====================================")
    print("=====================================")
    print("{} | Finished after {} seconds.".format(verbose, t2-t1))

    return results, times





