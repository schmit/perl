from collections import namedtuple
import random

from ...priors import NormalPrior, DirichletPrior
from ...mdp import MDP, value_iteration, find_all_states
from ...rl.environment import mdp_to_env
from .core import Algorithm

Posterior = namedtuple("Posterior", "transitions rewards indexer")
Indexer = namedtuple("Indexer", "from_index to_index")

def create_prior(states, actions, transitions, p_reward = lambda: NormalPrior(0, 1, 1)):
    """
    Creates a prior for an MDP with <states> states and <actions> actions,
    based on <transitions> in an attempt to copy the structure of the MDP

    Args:
        - states: list with all states
        - actions: function(state) -> [actions]
        - transitions: function(state, action) -> [(prob, (new_state, reward))]
        - p_reward: prior for reward distribution (default: Normal(0, 1, 1))

    returns:
        Posterior object
    """
    def reachable_states(state, action):
        """ Return states reachable from (state, action)-pair """
        return frozenset(new_state
                for _, (new_state, _) in transitions(state, action))

    states = sorted(states)
    nstates = len(states)

    from_index = {}
    to_index = {}
    indexer = Indexer(from_index, to_index)

    trans = {}

    for state in states:
        for action in actions(state):
            reachable = reachable_states(state, action)
            trans[(state, action)] = DirichletPrior(len(reachable), 1/(len(reachable) + 2))
            for idx, new_state in enumerate(reachable):
                from_index[(state, action, idx)] = new_state
                to_index[(state, action, new_state)] = idx

    rewards = {(state, action): p_reward()
            for state in states for action in actions(state)}

    return Posterior(trans, rewards, indexer)

def sample_posterior(posterior):
    """
    Samples transitions and rewards from posterior
    """
    transitions = {(state, action): post.sample()
            for (state, action), post in posterior.transitions.items()}
    rewards = {(state, action): post.sample()
            for (state, action), post in posterior.rewards.items()}
    return transitions, rewards

def get_map(posterior):
    """
    Return the MAP of the posterior
    """
    transitions = {(state, action): post.map
            for (state, action), post in posterior.transitions.items()}
    rewards = {(state, action): post.map
            for (state, action), post in posterior.rewards.items()}
    return transitions, rewards

def create_sampler(mdp, env, p_reward = lambda: NormalPrior(0, 1, 1), discount=0.97):
    """
    Create a sampler to sample MDPs from a posterior

    Args:
        - mdp: MDP to be learned
        - p_reward: prior over distributions of reward
    """
    # sorted so we can always recover the order from the indexer
    posterior = create_prior(env.states, mdp.actions, mdp.transitions, p_reward)
    sampler = lambda: sample_mdp(env, posterior, discount)

    return sampler, posterior

def sample_mdp(env, posterior, discount=0.97):
    """
    Sample an MDP from the posterior
    """
    sampled_transitions, sampled_rewards = sample_posterior(posterior)

    def transitions(state, action):
        # note do not consider the last sampled transition probability
        # as that is the probability to end the MDP
        return [(probability, (posterior.indexer.from_index[(state, action, idx)],
                     sampled_rewards[(state, action)]))
                for idx, probability in enumerate(sampled_transitions[(state, action)])]

    # make sure discount is less than one because sampled MDP might have
    # infinite cycles
    return MDP(env.initial_states, env.actions, transitions, min(discount, env.discount))

def map_mdp(env, posterior):
    """
    Return the MDP by taking the MAP for each element in the posterior
    """
    n_states = len(env.states)
    map_transitions, map_rewards = get_map(posterior)

    def transitions(state, action):
        # note do not consider the last sampled transition probability
        # as that is the probability to end the MDP
        # also, add a bit of noise to map_rewards to break ties
        return [(probability, (posterior.indexer.from_index[(state, action, idx)],
                     map_rewards[(state, action)] + random.random() * 1e-5))
                for idx, probability in enumerate(map_transitions[(state, action)])]

    # make sure discount is less than one because sampled MDP might have
    # infinite cycles
    return MDP(env.initial_states, env.actions, transitions, min(0.97, env.discount))

def update_posteriors(steps, posterior):
    """
    Take an episode in the form of <steps> and updates
    the <posterior> using the sars pairs (in-place)
    """
    for state, action, reward, new_state in steps:
        # update transition:
        posterior.transitions[(state, action)].update(posterior.indexer.to_index[(state, action, new_state)])
        # update reward
        posterior.rewards[(state, action)].update(reward)


class PosteriorSampling(Algorithm):
    def __init__(self,
            mdp,
            p_reward=lambda: NormalPrior(0, 1, 1),
            discount=0.95):
        self.env = mdp_to_env(mdp)
        self.sampler, self.posterior = create_sampler(
                mdp,
                self.env,
                p_reward,
                discount)

        self._updated_policy = False

    def init_episode(self):
        values, policy = value_iteration(self.sampler(), epsilon=1e-3)
        self.policy = policy

    def act(self, state):
        return self.policy[state]

    def learn(self, steps):
        update_posteriors(steps, self.posterior)
        self._updated_policy = False

    @property
    def optimal_policy(self):
        # cache optimal policy
        if not self._updated_policy:
            value, policy = value_iteration(map_mdp(self.env, self.posterior))
            self._opt_policy = policy
            self._updated_policy = True

        return self._opt_policy

    def __repr__(self):
        return "Posterior Sampling"

