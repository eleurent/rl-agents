import numpy as np

from rl_agents.agents.abstract import AbstractAgent


class ValueIterationAgent(AbstractAgent):
    def __init__(self, env, config=None):
        super(ValueIterationAgent, self).__init__(config)
        self.check_env(env)
        self.env = env

    @classmethod
    def default_config(cls):
        return dict(gamma=1.0)

    def act(self, state):
        return np.argmax(self.state_action_value()[state, :])

    def state_value(self, iterations=100):
        return ValueIterationAgent.fixed_point_iteration(
            np.zeros((self.env.observation_space.n,)),
            lambda v: ValueIterationAgent.value_from_action_values(self.bellman_equation(v)),
            iterations=iterations)

    def state_action_value(self, iterations=100):
        return ValueIterationAgent.fixed_point_iteration(
            self.env.reward.copy(),
            lambda q: self.bellman_equation(ValueIterationAgent.value_from_action_values(q)),
            iterations=iterations)

    @staticmethod
    def value_from_action_values(action_values):
        return action_values.max(axis=-1)

    def bellman_equation(self, value):
        return self.env.reward + self.config["gamma"] * \
               (self.env.transition * value.reshape((1, 1, self.env.observation_space.n))).sum(axis=-1)

    @staticmethod
    def fixed_point_iteration(initial, iterate, iterations):
        value = initial
        for _ in range(iterations):
            next_value = iterate(value)
            if np.allclose(value, next_value):
                break
            value = next_value
        return value

    @staticmethod
    def check_env(env):
        try:
            finite_mdp = __import__("finite_mdp")
            if not isinstance(env, finite_mdp.envs.finite_mdp.FiniteMDP):
                raise TypeError("Incorrect environment type")
            return True
        except (ModuleNotFoundError, TypeError):
            raise ValueError("This agent only supports environments of type finite_mdp.envs.FiniteMDP")

    def record(self, state, action, reward, next_state, done):
        pass

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()
