from collections import defaultdict
import numpy as np
import time

from perl.mdp.triangle import Triangle
from perl.mdp.chain import Chain

from perl.rl.algorithms import Qlearning, PosteriorSampling, TwoStepPosteriorSampling
from perl.rl.algorithms import TwoStepDecoupledPosteriorSampling, MaxVarPosteriorSampling
from perl.rl.algorithms import BOSS, Rmax
from perl.rl.distributed import run_distributed_sim, save_obj

num_cores = 3
mdp_number = 0
num_sims = 10 ; num_episodes = 50 ; log_every = 5

if mdp_number == 0:
    max_depth = 3 ; mdp = Triangle(max_depth)
    mdp_name = "Triangle-{}".format(max_depth)
else:
    n = 5 ; final_rew = 10 ; exploit_rew = 2 ; exit_prob=0.1
    mdp = Chain(n, final_rew, exploit_rew, exit_prob)
    mdp_name = "Chain-n={}".format(n)

QL = Qlearning ; PS = PosteriorSampling ; sTSPS = TwoStepPosteriorSampling
rTSPS = TwoStepDecoupledPosteriorSampling ; BS = BOSS ; rmax = Rmax ; mVar = MaxVarPosteriorSampling

algo_list = [QL, PS, sTSPS, rTSPS, BS, rmax, mVar]
algo_params = [{"mdp":mdp}, {"mdp":mdp}, {"mdp":mdp}, {"mdp":mdp}, {"mdp":mdp, "K":5, "B":3},
               {"mdp":mdp, "rmax_v":12, "K":4}, {"mdp":mdp, "q":4, "k":16}]
algo_names = ["QLearning", "PosteriorSampling", "sTSPS", "rTSPS", "BOSS", "Rmax", "VarMax"]

results = run_distributed_sim(mdp, algo_list, algo_names, algo_params, num_sims,
								num_episodes, log_every, num_cores)


name = "{}-numsim={}-numeps-{}".format(mdp_name, num_sims, num_episodes)
save_obj(results, name)



