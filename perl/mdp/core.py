from collections import namedtuple

"""
MDP is a collection of:

    initial_states -> [(prob, state)]
    actions(state) -> [action]
    transition(state, action) -> [(prob, (new_state, reward))]
    discount -> [0, 1]

    Note: new_state == None indicates end of episode
"""

MDP = namedtuple("MDP", "initial_states actions transitions discount")


def mean(reward):
    if isinstance(reward, int) or isinstance(reward, float):
        return reward
    return reward.mean

def find_all_states(mdp):
    states = {state for prob, state in mdp.initial_states()}
    queue = [state for state in states]
    while queue != []:
        state = queue.pop()
        for action in mdp.actions(state):
            for prob, (new_state, reward) in mdp.transitions(state, action):
                if new_state and new_state not in states:
                    states.add(new_state)
                    queue.append(new_state)

    return sorted(list(states))

def bellman(mdp, state, action, values):
    return sum(prob * (mean(reward) + mdp.discount * values.get(new_state, 0))
            for prob, (new_state, reward) in mdp.transitions(state, action))

def value_iteration(mdp, epsilon=1e-3, values=None):
    """ Solve MDP using value iteration

    Args:
        mdp: MDP instance
        epsilon (default: 1e-5): desired accuracy
        values (default: None): optional initial values

    Returns:
        values, policy: dictionaries with state->value and state->action
    """

    if values is None:
        values = {state: 0 for state in find_all_states(mdp)}

    iteration = 0
    while True:
        policy = {}
        new_values = {}
        # update all values
        for state, value in values.items():
            new_values[state], policy[state] = max((bellman(mdp, state, action, values), action)
                    for action in mdp.actions(state))

        # check for convergence
        diff = sum(abs(new_values[state] - values[state])
                for state in new_values.keys())
        if diff < epsilon:
            return new_values, policy
        else:
            values = new_values

        iteration += 1
        if iteration > 500:
            print("Warning: value_iteration not converged after 500 iterations")
            return new_values, policy

def policy_iteration(mdp, policy, epsilon=1e-3, values=None):
    """ Compute values of states when following <policy>

    Args:
        mdp: MDP instance
        policy: a policy
        epsilon (default: 1e-5): desired accuracy
        values (default: None): optional initial values

    Returns:
        values: dictionary with state->value
    """
    if values is None:
        values = {state: 0 for state in find_all_states(mdp)}

    while True:
        new_values = {}
        # update all values
        for state, value in values.items():
            new_values[state] = bellman(mdp, state, policy[state], values)

        # check for convergence
        diff = sum(abs(new_values[state] - values[state])
                for state in new_values.keys())
        if diff < epsilon:
            return new_values
        else:
            values = new_values



