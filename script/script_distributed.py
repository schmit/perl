from collections import defaultdict
import numpy as np
import time

from perl.mdp.blackjack import Blackjack
from perl.mdp.triangle import Triangle
from perl.mdp.numberline import Numberline
from perl.mdp.chain import Chain
from perl.mdp.sequence import Sequence, is_unique, is_increasing, most_duplicates
from perl.mdp.optimizer import Optimizer, f_sum, f_prod

from perl.rl.algorithms import Qlearning, PosteriorSampling, TwoStepPosteriorSampling
from perl.rl.algorithms import TwoStepDecoupledPosteriorSampling, MaxVarPosteriorSampling
from perl.rl.algorithms import BOSS, Rmax, rBOSS
from perl.rl.distributed import run_distributed_sim, save_obj

num_cores = 3
mdp_number = 4
num_sims = 9 ; num_episodes = 30 ; log_every = 3

if mdp_number == 0:
    max_depth = 3 ; mdp = Triangle(max_depth)
    mdp_name = "Triangle-{}".format(max_depth)
elif mdp_number == 1:
    n = 5 ; final_rew = 10 ; exploit_rew = 2 ; exit_prob=0.1
    mdp = Chain(n, final_rew, exploit_rew, exit_prob)
    mdp_name = "Chain-n={}".format(n)
elif mdp_number == 2:
	k = 15 ; p_success = 0.9 ; discount = 1 ; n = 5
	win_condition = is_unique ; win_name = "unique"
	mdp = Sequence(win_condition, k=k, n=n, discount=discount, p_success=p_success)
	mdp_name = "Sequence-n{}-k{}-p{}-w={}".format(n, k, p_success, win_name)
elif mdp_number == 3:
	mdp = Blackjack()
	mdp_name = "Blackjack"
elif mdp_number == 4:
    n = 14 ; discount=1 ; p_success=0.75 ; p_die=0.05 ; p_left=0.6 ; p_right=0.7
    mdp = Numberline(n, discount, p_success, p_die, p_left, p_right)
    mdp_name = "Numberline-n{}-ps{}-pd{}-pl{}-pr{}".format(n, p_success, p_die, p_left, p_right)
else:
    f = f_sum ; fname = "fsum"
    xmin = -3 ; xmax = 5 ; ymin = 0 ; ymax = 0 ; p_random = 0.1 ; p_die=0.05 ; discount = 1
    mdp = Optimizer(f, xmin, xmax, ymin, ymax, p_random, p_die, discount)
    mdp_name = "Optimizer-f={}-xmin{}-xmax{}-ymin{}-ymax{}-prandom{}-pdie{}.".format(fname, xmin, xmax,
                                                                                     ymin, ymax, p_random, p_die)


QL = Qlearning ; PS = PosteriorSampling ; sTSPS = TwoStepPosteriorSampling
rTSPS = TwoStepDecoupledPosteriorSampling ; BS = BOSS ;
rmax = Rmax ; mVar = MaxVarPosteriorSampling ; rBS = rBOSS

rmax_v = 12
rmax_k = 4
boss_k = 5
boss_b = 3
varmax_q = 4
varmax_k = 16

algos = [(1, QL, {"mdp":mdp}, "QLearning"),
         (2, PS, {"mdp":mdp}, "PosteriorSampling"),
         (3, sTSPS, {"mdp":mdp}, "sTSPS"),
         (4, rTSPS, {"mdp":mdp}, "rTSPS"),
         (5, BS, {"mdp":mdp, "K":boss_k, "B":boss_b}, "BOSS(K{},B{})".format(boss_k, boss_b)),
         (6, rmax, {"mdp":mdp, "rmax_v":rmax_v, "K":rmax_k}, "Rmax({},K{})".format(rmax_v, rmax_k)), 
         (7, mVar, {"mdp":mdp, "rmax_v":rmax_v, "K":rmax_k}, "VarMax(q{},k{})".format(varmax_q, varmax_k)),
         (8, rBS, {"mdp":mdp, "K":boss_k, "B":boss_b}, "rBOSS(K{},B{})".format(boss_k, boss_b))]

algos_to_include = [1, 2, 3, 5, 6, 8]

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



