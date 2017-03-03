import math
from statistics import mean, stdev
import time
import random
from collections import defaultdict
import numpy as np
import pickle

# from joblib import Parallel, delayed
import multiprocessing
# from copy import deepcopy

from pathos.pools import ProcessPool, ThreadPool

from .algorithms import FixedPolicy
from ..mdp import policy_iteration, value_iteration
from .environment import env_value, mdp_to_env
from ..util import sample, sars
from .simulator import comparison_sim

def run_distributed_sim(mdp, algo_list, algo_names, algo_params, num_sims, num_episodes, log_every, num_cores):

    """
        Given an MDP and a list of algorithms, a number of simulations are equally distributed across cores.
    """

    num_cores = min(multiprocessing.cpu_count(), num_cores)
    print("Using {} cores for parallel processing.".format(num_cores))

    sims_core = int(num_sims/num_cores)

    # mdp, algo_list, algo_names, algo_params, num_sims, num_episodes, log_every

    arg_base = {"mdp":mdp, "algo_list":algo_list, "algo_names":algo_names, "algo_params":algo_params,
               "num_sims":num_sims, "num_episodes":num_episodes, "log_every":log_every}
    arg_maps = [{k:v for k,v in arg_base.items()} for i in range(num_cores)]
    for i in range(num_cores):
        arg_maps[i]["worker"] = i

    pool = ProcessPool(num_cores)
    results_list = pool.map(process_simulation, arg_maps)

    # results_list = Parallel(n_jobs=num_cores)(delayed(process_simulation)(i, by_parts, algo_list,
    #    algo_names, algo_params, sims_core, num_episodes, log_every) for i in range(num_cores))

    results = defaultdict(list)
    for elt in results_list:
        for key, val in elt.items():
            results[key].append(val)

    return results

# def process_simulation(i, mdp, algo_list, algo_names, algo_params, num_sims, num_episodes, log_every):
def process_simulation(arg_dict):
    """
    This function is run in parallel across diferent nodes. All results are later combined.
    """

    worker = arg_dict["worker"]
    mdp = arg_dict["mdp"]
    algo_list = arg_dict["algo_list"]
    algo_names = arg_dict["algo_names"]
    algo_params = arg_dict["algo_params"]
    num_sims = arg_dict["num_sims"]
    num_episodes = arg_dict["num_episodes"]
    log_every = arg_dict["log_every"]

    np.random.seed(worker * int(time.time() % 100000))
    random.seed(worker * int(5 * time.time() % 100000))

    results, times = comparison_sim(mdp, algo_list, algo_names, algo_params,
                    num_sims=num_sims, num_episodes=num_episodes, log_every=log_every, verbose=False)

    return results

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 0)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

