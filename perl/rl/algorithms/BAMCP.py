import random
import statistics
import numpy as np
import copy
from collections import defaultdict

from ...priors import NormalPrior
from ..environment import mdp_to_env, env_value, sample_reward
from ..simulator import episode
from ...mdp import policy_iteration, value_iteration, imp_value_iteration
from .posteriorsampling import PosteriorSampling, create_sampler, update_posteriors, Posterior, sample_mdp, Indexer
from .core import FixedPolicy
from ...util import sample, sars

class BAMCP(PosteriorSampling):
    def __init__(self,
            mdp,
            p_reward=lambda: NormalPrior(0, 1, 1),
            discount=0.95,
            iter_budget=4,
            c=10):

        self.env = mdp_to_env(mdp)
        self.sampler, self.posterior = create_sampler(
                mdp,
                self.env,
                p_reward,
                discount)

        self.latest_values = None
        self._updated_policy = False
        self._training_logs = []

        self.seen_episodes = 0

        self.iter_budget = iter_budget
        self.c = c

        self.state_visits = defaultdict(float)
        self.action_visits = defaultdict(float)
        self.q_value = defaultdict(float)

        self.num_states = len(self.env.states)
        self.num_actions = np.max([len(self.env.actions(s)) for s in self.env.states])

        self.history_t = np.zeros((self.num_states, self.num_states))
        self.history_r = np.zeros((self.num_states, self.num_actions))

    def learn(self, steps):
        super(BAMCP, self).learn(steps)
        for state, action, reward, new_state in steps:
            self.history_t[state, new_state] += 1
            self.history_r[state, action] += reward

    def init_episode(self):

        self.seen_episodes += 1

        i_bel = self.extract_belief_state(self.history_t, self.history_r)

        for i in range(self.iter_budget):
            mdp = self.sampler()
            i_state = sample(self.env.initial_states())
            self.simulate((i_state, i_bel), mdp, 0)

        self.policy = {s:np.random.randint(len(self.actions(s))) for s in self.env.states}
        
        updated_scores = defaultdict(list)
        for key, val in self.q_value.items():
            v = key[:-1]
            action = key[-1]
            state, bel = self.decode_ext(v)
            updated_scores[state].append((val, action))
        for key, vals in updated_scores.items():
            self.policy[key] = sorted(vals, key=lambda elt: elt[0], reverse=True)[0][1]

        return None

    def simulate(self, ext_state, mdp, d):

        state, bel = ext_state
        v = self.encode_ext(ext_state)

        if state is None or d > 20:
            return 0

        if v not in self.state_visits:
            # initialize to zero
            self.state_visits[v] = 0
            for a in mdp.actions(state):
                self.q_value[v + tuple([a])] = 0
                self.action_visits[v + tuple([a])] = 0
            # run rollout
            action = self.rollout_policy(ext_state, len(mdp.actions(state)))
            next_state, reward = self.sample_transition(state, action, mdp)
            next_ext_state = self.update_belief_state(ext_state, action, next_state, reward)
            rollout_r = mdp.discount * self.rollout(next_ext_state, mdp, d+1)
            # update the lookahead tree
            self.state_visits[v] = 1
            self.action_visits[v + tuple([a])] = 1
            self.q_value[v + tuple([a])] = rollout_r
            return rollout_r
        else:
            action = np.argmax((self.q_value[v + tuple([a])] + self.c * np.sqrt(np.log(self.state_visits[v]) / self.action_visits[v + tuple([a])])
                                for a in mdp.actions(state)))
            next_state, reward = self.sample_transition(state, action, mdp)
            next_ext_state = self.update_belief_state(ext_state, action, next_state, reward)
            simulate_r = mdp.discount * self.simulate(next_ext_state, mdp, d+1)
            self.state_visits[v] += 1
            self.action_visits[v + tuple([a])] += 1
            self.q_value[v + tuple([a])] += (simulate_r - self.q_value[v + tuple([a])]) / self.action_visits[v + tuple([a])]
            return simulate_r

    def sample_transition(self, state, action, mdp):
        print(state)
        print(action)
        new_state, reward_distribution = sample(mdp.transitions(state, action))
        reward = sample_reward(reward_distribution)
        return new_state, reward

    def rollout_policy(self, ext_state, num_actions):
        return np.random.randint(num_actions)

    def rollout(self, ext_state, mdp, d):
        state, bel = ext_state
        if state is None or d > 20:
            return 0
        action = self.rollout_policy(ext_state, len(mdp.actions(state)))
        next_state, reward = self.sample_transition(state, action, mdp)
        next_ext_state = self.extend_state(ext_state, next_state, reward)
        return reward + mdp.discount * self.rollout(next_ext_state, mdp, d+1)

    def extract_belief_state(self, trans, rewards):
        return (trans.copy(), rewards.copy())

    def update_belief_state(self, ext_state, action, next_state, reward):
        state, belief = ext_state
        trans, rew = belief
        new_trans = trans.copy()
        new_rew = rew.copy()
        new_trans[state, next_state] += 1
        new_rew[state, action] += reward
        return (next_state, (new_trans, new_rew))

    def encode_ext(self, ext_state):
        state, belief = ext_state
        v = [state]
        trans, rew = belief
        n, m = trans.shape
        v += trans.reshape(n*m,).tolist()
        n, m = rew.shape
        v += rew.reshape(n*m,).tolist()
        return tuple(v)

    def decode_ext(self, v):
        state = v[0]
        trans = v[1:1+self.num_states*self.num_states]
        trans = np.array(trans).reshape(self.num_states, self.num_states)
        rew = v[1+self.num_states*self.num_states:]
        rew = np.array(rew).reshape(self.num_states, self.num_actions)
        return state, (trans, rew)

    def __repr__(self):
        return "BAMCP-Sampling"

