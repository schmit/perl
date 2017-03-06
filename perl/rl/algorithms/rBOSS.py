import random
import statistics
import numpy as np
from collections import defaultdict

from ...priors import NormalPrior
from ..environment import mdp_to_env, env_value
from ..memory import RingBuffer
from ...mdp import MDP, policy_iteration, value_iteration
from .posteriorsampling import PosteriorSampling, create_sampler

class rBOSS(PosteriorSampling):
	"""
		Implements repeated Best of Sampled Set (BOSS) Algorithm.
		https://arxiv.org/pdf/1205.2664.pdf
	"""
	def __init__(self,
			mdp,
			p_reward=lambda: NormalPrior(0, 1, 1),
			discount=0.95,
			K=4.0, # number of MDPs sampled at each update
			B=10): # number of visits for (s,a) before update

		self.env = mdp_to_env(mdp)
		self.sampler, self.posterior = create_sampler(
				mdp,
				self.env,
				p_reward,
				discount)

		self.K = K
		self.B = B

		self.latest_values = None
		self._updated_policy = False
		self.visits_sa = defaultdict(lambda: 0)
		self.do_sample = True

	def init_episode(self):

		if self.do_sample:
			mdp_set = [self.sampler() for _ in range(self.K)]
			mdp_merged = merge_mdps(mdp_set)
			value_merged, policy_merged = value_iteration(mdp_merged, epsilon=1e-3,
														  values=self.latest_values)
			self.policy = policy_projection(policy_merged)
			self.do_sample = False

		return None

	def learn(self, steps):
		# update_posteriors(steps, self.posterior)
		# self._updated_policy = False
		super(BOSS, self).learn(steps)
		# steps are (state, action, reward, new_state) tuples
		for s, a, r, s2 in steps:
			self.visits_sa[(s,a)] += 1
			if self.visits_sa[(s,a)] % self.B == 0 and self.visits_sa[(s,a)] >= self.B:
				# sample every B visits to (s,a)
				self.do_sample = True

	def __repr__(self):
		return "rBOSS Sampling"


def merge_mdps(mdp_list):
	""" merge_mdps
		Given a list of MDPs with common state and action spaces,
		create an MDP with same state space, but actions a_{i,j}(s)
		for action i on mdp j at state s, that lead to transitions
		and rewards in MDP i: P_j(s'|s,a_i), R_j(s,a_i).
		"""
	if len(mdp_list) == 0:
		raise ValueError('Cannot merge empty list of MDPs.')

	def initial_states():
		return mdp_list[0].initial_states()

	def actions(state):
		action_space = []
		for j, mdp in enumerate(mdp_list):
			action_space += [(i,j) for i in mdp.actions(state)]
		return action_space

	def transitions(state, action):
		i, j = action
		return mdp_list[j].transitions(state, i)

	return MDP(initial_states, actions, transitions, mdp_list[0].discount)

def policy_projection(policy):
	""" The current policy goes from state to pairs (action, mdp).
		Just remove the mdp from the pair."""
	return {s:action for s, (action, mdp) in policy.items()}
