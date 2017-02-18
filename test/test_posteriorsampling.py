from perl.mdp.numberline import Numberline
from perl.mdp import value_iteration

from perl.rl.environment import mdp_to_env, env_value

from perl.rl.algorithms import PosteriorSampling, TwoStepPosteriorSampling
from perl.rl.simulator import episode, live

from perl.mdp import find_all_states
from perl.rl.algorithms.posteriorsampling import create_prior, sample_mdp

mdp = Numberline(4)
mdp_states = find_all_states(mdp)

def test_sampler_same_states():
    PS = PosteriorSampling(mdp)
    assert mdp_states == find_all_states(PS.sampler())

def test_sampler_same_transitions():
    PS = PosteriorSampling(mdp)
    sampled_mdp = PS.sampler()

    for state in mdp_states:
        for action in mdp.actions(state):
            reachable_original = frozenset(new_state
                    for _, (new_state, _) in mdp.transitions(state, action))
            reachable_sampled = frozenset(new_state
                    for _, (new_state, _) in sampled_mdp.transitions(state, action))
            assert reachable_original == reachable_sampled

def test_rewards():
    PS = PosteriorSampling(mdp)
    posterior_rewards = PS.posterior.rewards
    for state in mdp_states:
        for action in mdp.actions(state):
            assert (state, action) in posterior_rewards

