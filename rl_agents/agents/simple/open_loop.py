import copy

from rl_agents.agents.abstract import AbstractAgent


class OpenLoopAgent(AbstractAgent):
    """
        Execute a given list of actions
    """

    def __init__(self, env, config=None):
        super(OpenLoopAgent, self).__init__(config)
        self.actions = self.config["actions"]

    @classmethod
    def default_config(cls):
        return dict(actions=[],
                    default_action=0)

    def plan(self, state):
        if self.actions:
            actions = copy.deepcopy(self.actions)
            self.actions.pop(0)
            return actions
        else:
            return [self.config["default_action"]]

    def act(self, state):
        return self.plan(state)[0]

    def seed(self, seed=None):
        return None

    def reset(self):
        self.actions = self.config["actions"]

    def record(self, state, action, reward, next_state, done, info):
        pass

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()

