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
from perl.mdp.stockprecomp import StockPreComp

from perl.rl.algorithms import Qlearning, PosteriorSampling, TwoStepPosteriorSampling
from perl.rl.algorithms import TwoStepDecoupledPosteriorSampling, MaxVarPosteriorSampling, MinVarPosteriorSampling
from perl.rl.algorithms import BOSS, Rmax, rBOSS, InfoMaxSampling, InfoImprovSampling, MCfGreedySampling
from perl.rl.simulator import reward_path, comparison_sim

from perl.priors import BetaPrior
from collections import defaultdict

from perl.rl.distributed import run_distributed_sim, save_obj

num_cores = 50

mdp_number = 6 ; num_sims = num_cores * 3 ; num_episodes = 100 ; log_every = 4

if mdp_number == 0:
    max_depth = 3 ; mdp = Triangle(max_depth, probs=[0.3 + 0.01*i for i in range(41)])
    mdp_name = "Triangle-{}".format(max_depth)
    binary_reward = 1
elif mdp_number == 1:
    n = 7 ; final_rew = 14 ; exploit_rew = 1 ; exit_prob=0.1 ; slip_prob = 0.05
    mdp = Chain(n, final_rew, exploit_rew, exit_prob, slip_prob)
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
    max_depth = 6 ; means=[0.5, 1, 1.25, 1.5, 1.75, 2, 3] ; sigma2=[1]
    mdp = NormalTriangle(max_depth, means, sigma2)
    mdp_name = "Normal-Triangle-{}-means-{}".format(max_depth, means)
    binary_reward = 0
elif mdp_number == 7:
    max_depth = 5 ; means=[0.5, 1, 1.25, 1.5, 1.75, 2, 3] ; sigma2=1
    mdp_b = NormalTriangle(max_depth, means, sigma2)
    mdp_name_b = "Normal-Triangle-{}-means-{}".format(max_depth, means)
    c = 1 ; sigma2 = 1
    mdp = InfoStore(mdp_b, c, sigma2)
    mdp_name = "InfoStore-c={}-s2={}-b={}".format(c, sigma2, mdp_name_b)
    binary_reward = 0
elif mdp_number == 8:
    n = 5 ; m = 3 ; p_c = 0.05 ; p_die = 0.1 ; discount = 1
    mdp = Retention(n, m, p_c, p_die, discount)
    mdp_name = "Retention-n{}-m{}-pc{}-pd{}".format(n, m, p_c, p_die)    
    binary_reward = 1
else:
    # M = market transition matrix.
    # D = market_state x num_incoming_cust probability matrix.
    # P = market_state x prices = prob of buying in the market state at price p
    price_set = [1, 2, 3, 4, 5] ; S=12 ; T=5
    M = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.80]]) # market transition
    D = np.array([[0.3, 0.2, 0.15, 0.15, 0.1, 0.075, 0.025, 0],   # max num cust = 7
                  [0.1, 0.20, 0.25, 0.15, 0.15, 0.05, 0.05, 0.05],
                  [0.05, 0.15, 0.30, 0.10, 0.20, 0.075, 0.075, 0.05]])
    P = np.array([[0.7, 0.6, 0.4, 0.2, 0.1],
                 [0.7, 0.6, 0.4, 0.3, 0.2],
                 [0.85, 0.5, 0.45, 0.25, 0.25]])
    mdp = StockPreComp(M, D, P, S, T, price_set)
    mdp_name = "Stock-S{}-T{}".format(S, T)
    binary_reward = 0

QL = Qlearning ; PS = PosteriorSampling ; sTSPS = TwoStepPosteriorSampling
rTSPS = TwoStepDecoupledPosteriorSampling ; BS = BOSS ;
rmax = Rmax ; maxVar = MaxVarPosteriorSampling ; minVar = MinVarPosteriorSampling ; rBS = rBOSS
infoMax = InfoMaxSampling ; infoImp = InfoImprovSampling ; MCfgreedy = MCfGreedySampling

rmax_v = 12
rmax_k = 4
boss_k = 10
boss_b = 1
varmax_q = 4
varmax_k = 16
varmin_q = 4
varmin_k = 16

infomax_q = 8    # num policies
infomax_k = 12    # num mdps
infomax_episdata = 5
infomax_total_budget = num_episodes # do PS with prob
infoimp_alpha = 1

# MCfGreedy Functions
f1 = lambda vals: np.mean(vals) ; f1_name = "mean"
f2 = lambda vals: np.mean(vals) + np.std(vals) ; f2_name = "1std"
f3 = lambda vals: np.mean(vals) + 2 * np.std(vals) ; f3_name = "2std"
f4 = lambda vals: np.mean(vals) + 3 * np.std(vals) ; f4_name = "3std"
f5 = lambda vals: np.max(vals) ; f5_name = "max"
f6 = lambda vals: np.max(vals) ; f6_name = "min"
f7 = lambda vals: np.var(vals) ; f7_name = "var"
f8 = lambda vals: -np.var(vals) ; f8_name = "-var"

if binary_reward:
    algos = [(1, QL, {"mdp":mdp}, "QLearning"),
             (2, PS, {"mdp":mdp, "p_reward":lambda: BetaPrior(1, 1)}, "PosteriorSampling"),
             (3, sTSPS, {"mdp":mdp, "p_reward":lambda: BetaPrior(1, 1)}, "sTSPS"),
             (4, rTSPS, {"mdp":mdp, "p_reward":lambda: BetaPrior(1, 1)}, "rTSPS"),
             (5, BS, {"mdp":mdp, "K":boss_k, "B":boss_b, "p_reward":lambda: BetaPrior(1, 1)}, "BOSS(K{},B{})".format(boss_k, boss_b)),
             (6, rmax, {"mdp":mdp, "rmax_v":rmax_v, "K":rmax_k}, "Rmax({},K{})".format(rmax_v, rmax_k)), 
             (7, maxVar, {"mdp":mdp, "k":varmax_k, "q":varmax_q, "p_reward":lambda: BetaPrior(1, 1)}, "VarMax(q{},k{})".format(varmax_q, varmax_k)),
             (8, minVar, {"mdp":mdp, "k":varmin_q, "q":varmin_k, "p_reward":lambda: BetaPrior(1, 1)}, "VarMin(q{},k{})".format(varmin_q, varmin_k)),
             (9, infoMax, {"mdp":mdp, "k":infomax_k, "q":infomax_q, "p_reward":lambda: BetaPrior(1, 1), "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "InfoMax(q{},k{})".format(infomax_q, infomax_k)),
             (10, rBS, {"mdp":mdp, "K":boss_k, "B":boss_b, "p_reward":lambda: BetaPrior(1, 1)}, "rBOSS(K{},B{})".format(boss_k, boss_b)),
             (11, infoImp, {"mdp":mdp, "alpha":infoimp_alpha, "k":infomax_k, "q":infomax_q, "p_reward":lambda: BetaPrior(1, 1), "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "InfoImp(q{},k{},alpha{})".format(infomax_q, infomax_k, infoimp_alpha)),
             (12, MCfgreedy, {"mdp":mdp, "f":f1, "f_name":f1_name, "k":infomax_k, "q":infomax_q, "p_reward":lambda: BetaPrior(1, 1), "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f1_name)),
             (13, MCfgreedy, {"mdp":mdp, "f":f2, "f_name":f2_name, "k":infomax_k, "q":infomax_q, "p_reward":lambda: BetaPrior(1, 1), "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f2_name)),
             (14, MCfgreedy, {"mdp":mdp, "f":f3, "f_name":f3_name, "k":infomax_k, "q":infomax_q, "p_reward":lambda: BetaPrior(1, 1), "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f3_name)),
             (15, MCfgreedy, {"mdp":mdp, "f":f4, "f_name":f4_name, "k":infomax_k, "q":infomax_q, "p_reward":lambda: BetaPrior(1, 1), "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f4_name)),
             (16, MCfgreedy, {"mdp":mdp, "f":f5, "f_name":f5_name, "k":infomax_k, "q":infomax_q, "p_reward":lambda: BetaPrior(1, 1), "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f5_name)),
             (17, MCfgreedy, {"mdp":mdp, "f":f6, "f_name":f6_name, "k":infomax_k, "q":infomax_q, "p_reward":lambda: BetaPrior(1, 1), "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f6_name)),
             (18, MCfgreedy, {"mdp":mdp, "f":f7, "f_name":f7_name, "k":infomax_k, "q":infomax_q, "p_reward":lambda: BetaPrior(1, 1), "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f7_name)),
             (19, MCfgreedy, {"mdp":mdp, "f":f8, "f_name":f8_name, "k":infomax_k, "q":infomax_q, "p_reward":lambda: BetaPrior(1, 1), "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f8_name))]

else:
    algos = [(1, QL, {"mdp":mdp}, "QLearning"),
             (2, PS, {"mdp":mdp}, "PosteriorSampling"),
             (3, sTSPS, {"mdp":mdp}, "sTSPS"),
             (4, rTSPS, {"mdp":mdp}, "rTSPS"),
             (5, BS, {"mdp":mdp, "K":boss_k, "B":boss_b}, "BOSS(K{},B{})".format(boss_k, boss_b)),
             (6, rmax, {"mdp":mdp, "rmax_v":rmax_v, "K":rmax_k}, "Rmax({},K{})".format(rmax_v, rmax_k)), 
             (7, maxVar, {"mdp":mdp, "k":varmax_k, "q":varmax_q}, "VarMax(q{},k{})".format(varmax_q, varmax_k)),
             (8, minVar, {"mdp":mdp, "k":varmin_q, "q":varmin_k}, "VarMin(q{},k{})".format(varmin_q, varmin_k)),
             (9, infoMax, {"mdp":mdp, "k":infomax_k, "q":infomax_q, "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "InfoMax(q{},k{})".format(infomax_q, infomax_k)),
             (10, rBS, {"mdp":mdp, "K":boss_k, "B":boss_b}, "rBOSS(K{},B{})".format(boss_k, boss_b)),
             (11, infoImp, {"mdp":mdp, "alpha":infoimp_alpha, "k":infomax_k, "q":infomax_q, "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "InfoImp(q{},k{},alpha{})".format(infomax_q, infomax_k, infoimp_alpha)),
             (12, MCfgreedy, {"mdp":mdp, "f":f1, "f_name":f1_name, "k":infomax_k, "q":infomax_q, "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f1_name)),
             (13, MCfgreedy, {"mdp":mdp, "f":f2, "f_name":f2_name, "k":infomax_k, "q":infomax_q, "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f2_name)),
             (14, MCfgreedy, {"mdp":mdp, "f":f3, "f_name":f3_name, "k":infomax_k, "q":infomax_q, "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f3_name)),
             (15, MCfgreedy, {"mdp":mdp, "f":f4, "f_name":f4_name, "k":infomax_k, "q":infomax_q, "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f4_name)),
             (16, MCfgreedy, {"mdp":mdp, "f":f5, "f_name":f5_name, "k":infomax_k, "q":infomax_q, "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f5_name)),
             (17, MCfgreedy, {"mdp":mdp, "f":f6, "f_name":f6_name, "k":infomax_k, "q":infomax_q, "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f6_name)),
             (18, MCfgreedy, {"mdp":mdp, "f":f7, "f_name":f7_name, "k":infomax_k, "q":infomax_q, "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f7_name)),
             (19, MCfgreedy, {"mdp":mdp, "f":f8, "f_name":f8_name, "k":infomax_k, "q":infomax_q, "num_epis_data":infomax_episdata, "total_budget":infomax_total_budget}, "MCf(q{},k{},f{})".format(infomax_q, infomax_k, f8_name))]

algos_to_include = [1, 2, 9, 10, 12, 14, 16, 17, 19]

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



