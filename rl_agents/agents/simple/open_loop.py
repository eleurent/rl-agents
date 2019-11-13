from rl_agents.agents.common.abstract import AbstractAgent


class OpenLoopAgent(AbstractAgent):
    """
        Execute a given list of actions
    """

    def __init__(self, env, config=None):
        super(OpenLoopAgent, self).__init__(config)
        self.env = env
        self.actions = None
        self.reset()
        self.default_horizon = 1

    @classmethod
    def default_config(cls):
        return dict(actions=[],
                    default_action=0)

    def plan(self, state):
        if self.actions:
            self.actions.pop(0)
        return self.get_plan()

    def get_plan(self):
        if self.actions:
            return self.actions.copy()
        else:
            return [self.config["default_action"]] * self.default_horizon

    def act(self, state):
        return self.plan(state)[0]

    def seed(self, seed=None):
        return None

    def reset(self):
        self.actions = [None] + self.config["actions"]

    def record(self, state, action, reward, next_state, done, info):
        pass

    def save(self, filename):
        return False

    def load(self, filename):
        return False

