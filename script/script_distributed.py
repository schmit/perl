from collections import defaultdict
import numpy as np
import time

from perl.mdp import find_all_states, value_iteration, policy_iteration
from perl.rl.environment import mdp_to_env, env_value

from perl.mdp.blackjack import Blackjack
from perl.mdp.triangle import Triangle
from perl.mdp.numberline import Numberline
from perl.mdp.chain import Chain
from perl.mdp.sequence import Sequence, is_unique, is_increasing, most_duplicates
from perl.mdp.optimizer import Optimizer, f_sum, f_prod
from perl.mdp.normal_triangle import NormalTriangle
from perl.mdp.infostore import InfoStore

from perl.rl.algorithms import Qlearning, PosteriorSampling, TwoStepPosteriorSampling
from perl.rl.algorithms import TwoStepDecoupledPosteriorSampling, MaxVarPosteriorSampling, MinVarPosteriorSampling
from perl.rl.algorithms import BOSS, Rmax, rBOSS, InfoMaxSampling
from perl.rl.simulator import reward_path, comparison_sim

from perl.priors import BetaPrior
from collections import defaultdict

from perl.rl.distributed import run_distributed_sim, save_obj

num_cores = 45

mdp_number = 6 ; num_sims = num_cores * 4 ; num_episodes = 60 ; log_every = 3

if mdp_number == 0:
    max_depth = 15 ; mdp = Triangle(max_depth, probs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    mdp_name = "Triangle-{}".format(max_depth)
    binary_reward = 1
elif mdp_number == 1:
    n = 7 ; final_rew = 0.7 ; exploit_rew = 0.05 ; exit_prob=0.1
    mdp = Chain(n, final_rew, exploit_rew, exit_prob)
    mdp_name = "Chain-n={}".format(n)
    binary_reward = 0
elif mdp_number == 2:
    k = 15 ; p_success = 0.9 ; discount = 1 ; n = 5
    win_condition = is_unique ; win_name = "unique"
    mdp = Sequence(win_condition, k=k, n=n, discount=discount, p_success=p_success)
    mdp_name = "Sequence-n{}-k{}-p{}-w={}".format(n, k, p_success, win_name)
    binary_reward = 0
elif mdp_number == 3:
    mdp = Blackjack()
    mdp_name = "Blackjack"
    binary_reward = 1
elif mdp_number == 4:
    n = 14 ; discount=1 ; p_success=0.75 ; p_die=0.05 ; p_left=0.6 ; p_right=0.7
    mdp = Numberline(n, discount, p_success, p_die, p_left, p_right)
    mdp_name = "Numberline-n{}-ps{}-pd{}-pl{}-pr{}".format(n, p_success, p_die, p_left, p_right)
    binary_reward = 0
elif mdp_number == 5:
    f = f_sum ; fname = "fsum"
    xmin = -3 ; xmax = 3 ; ymin = -3 ; ymax = 3 ; p_random = 0.1 ; p_die=0.05 ; discount = 1
    mdp = Optimizer(f, xmin, xmax, ymin, ymax, p_random, p_die, discount)
    mdp_name = "Optimizer-f={}-xmin{}-xmax{}-ymin{}-ymax{}-prandom{}-pdie{}".format(fname, xmin, xmax,
                                                                                     ymin, ymax, p_random, p_die)
    binary_reward = 0
elif mdp_number == 6:
    max_depth = 15 ; means=[0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3] ; sigma2=1
    mdp = NormalTriangle(max_depth, means, sigma2)
    mdp_name = "Normal-Triangle-{}-means-{}".format(max_depth, means)
    binary_reward = 0
elif mdp_number == 7:
    max_depth = 12 ; means=[0.5, 1, 1.25, 1.5, 1.75, 2, 3] ; sigma2=1
    mdp_b = NormalTriangle(max_depth, means, sigma2)
    mdp_name_b = "Normal-Triangle-{}-means-{}".format(max_depth, means)
    c = 1 ; sigma2 = 1
    mdp = InfoStore(mdp_b, c, sigma2)
    mdp_name = "InfoStore-c={}-s2={}-b={}".format(c, sigma2, mdp_name_b)
    binary_reward = 0
else:
    n = 5 ; m = 3 ; p_c = 0.05 ; p_die = 0.1 ; discount = 1
    mdp = Retention(n, m, p_c, p_die, discount)
    mdp_name = "Retention-n{}-m{}-pc{}-pd{}".format(n, m, p_c, p_die)    
    binary_reward = 1

QL = Qlearning ; PS = PosteriorSampling ; sTSPS = TwoStepPosteriorSampling
rTSPS = TwoStepDecoupledPosteriorSampling ; BS = BOSS ;
rmax = Rmax ; maxVar = MaxVarPosteriorSampling ; minVar = MinVarPosteriorSampling ; rBS = rBOSS
infoMax = InfoMaxSampling

rmax_v = 12
rmax_k = 4
boss_k = 10
boss_b = 1
varmax_q = 4
varmax_k = 16
varmin_q = 4
varmin_k = 16

infomax_q = 20    # num policies
infomax_k = 20    # num mdps
infomax_episdata = 5

if binary_reward:
    algos = [(1, QL, {"mdp":mdp}, "QLearning"),
             (2, PS, {"mdp":mdp, "p_reward":lambda: BetaPrior(1, 1)}, "PosteriorSampling"),
             (3, sTSPS, {"mdp":mdp, "p_reward":lambda: BetaPrior(1, 1)}, "sTSPS"),
             (4, rTSPS, {"mdp":mdp, "p_reward":lambda: BetaPrior(1, 1)}, "rTSPS"),
             (5, BS, {"mdp":mdp, "K":boss_k, "B":boss_b, "p_reward":lambda: BetaPrior(1, 1)}, "BOSS(K{},B{})".format(boss_k, boss_b)),
             (6, rmax, {"mdp":mdp, "rmax_v":rmax_v, "K":rmax_k}, "Rmax({},K{})".format(rmax_v, rmax_k)), 
             (7, maxVar, {"mdp":mdp, "k":varmax_k, "q":varmax_q, "p_reward":lambda: BetaPrior(1, 1)}, "VarMax(q{},k{})".format(varmax_q, varmax_k)),
             (8, minVar, {"mdp":mdp, "k":varmin_q, "q":varmin_k, "p_reward":lambda: BetaPrior(1, 1)}, "VarMin(q{},k{})".format(varmin_q, varmin_k)),
             (9, infoMax, {"mdp":mdp, "k":infomax_k, "q":infomax_q, "p_reward":lambda: BetaPrior(1, 1), "num_epis_data":infomax_episdata}, "InfoMax(q{},k{})".format(infomax_q, infomax_k)),
             (10, rBS, {"mdp":mdp, "K":boss_k, "B":boss_b, "p_reward":lambda: BetaPrior(1, 1)}, "rBOSS(K{},B{})".format(boss_k, boss_b))]
else:
    algos = [(1, QL, {"mdp":mdp}, "QLearning"),
             (2, PS, {"mdp":mdp}, "PosteriorSampling"),
             (3, sTSPS, {"mdp":mdp}, "sTSPS"),
             (4, rTSPS, {"mdp":mdp}, "rTSPS"),
             (5, BS, {"mdp":mdp, "K":boss_k, "B":boss_b}, "BOSS(K{},B{})".format(boss_k, boss_b)),
             (6, rmax, {"mdp":mdp, "rmax_v":rmax_v, "K":rmax_k}, "Rmax({},K{})".format(rmax_v, rmax_k)), 
             (7, maxVar, {"mdp":mdp, "k":varmax_k, "q":varmax_q}, "VarMax(q{},k{})".format(varmax_q, varmax_k)),
             (8, minVar, {"mdp":mdp, "k":varmin_q, "q":varmin_k}, "VarMin(q{},k{})".format(varmin_q, varmin_k)),
             (9, infoMax, {"mdp":mdp, "k":infomax_k, "q":infomax_q, "num_epis_data":infomax_episdata}, "InfoMax(q{},k{})".format(infomax_q, infomax_k)),
             (10, rBS, {"mdp":mdp, "K":boss_k, "B":boss_b}, "rBOSS(K{},B{})".format(boss_k, boss_b))]

algos_to_include = [1, 2, 9, 10]

algo_list = [elm[1] for elm in algos if elm[0] in algos_to_include]
algo_params = [elm[2] for elm in algos if elm[0] in algos_to_include]
algo_names = [elm[3] for elm in algos if elm[0] in algos_to_include]

print("Running Distributed Simulations. MDP: {}.".format(mdp_name))
print("Number of Simulations: {} with {} episodes | log every = {}.".format(num_sims, num_episodes, log_every))
print("Number of Cores: {}.".format(num_cores))
print("Algorithms: {}.".format(algo_names))

results = run_distributed_sim(mdp, algo_list, algo_names, algo_params, num_sims,
								num_episodes, log_every, num_cores)

results["mdp_name"] = mdp_name

name = "{}-numsim={}-numeps-{}_{}".format(mdp_name, num_sims, num_episodes, str(int(time.time())))
print(name)
save_obj(results, name)



