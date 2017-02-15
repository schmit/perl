

class Algorithm:
    """ Base class for RL algorithm """
    def init_episode(self):
        pass

    def act(self, state):
        raise NotImplementedError

    def learn(self, episode):
        pass


class FixedPolicy(Algorithm):
    def __init__(self, policy):
        self.policy = policy

    def act(self, state):
        return self.policy[state]


