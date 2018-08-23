import numpy as np

from rl_agents.agents.dynamic_programming.value_iteration import ValueIterationAgent


class RobustValueIterationAgent(ValueIterationAgent):
    def __init__(self, env, config=None):
        super(ValueIterationAgent, self).__init__(config)
        self.env = env
        self.transitions = np.array([])
        self.rewards = np.array([])
        self.models_from_config()

    @classmethod
    def default_config(cls):
        return dict(gamma=1.0,
                    models=[])

    def models_from_config(self):
        if not self.config["models"]:
            raise ValueError("No finite MDP model provided in agent configuration")
        self.transitions = np.array([mdp["transition"] for mdp in self.config["models"]])
        self.rewards = np.array([mdp["reward"] for mdp in self.config["models"]])

    def act(self, state):
        return np.argmax(self.state_action_value()[state, :])

    def state_value(self, iterations=100):
        return ValueIterationAgent.fixed_point_iteration(
            np.zeros((self.env.observation_space.n,)),
            lambda v: RobustValueIterationAgent.value_from_action_values(
                RobustValueIterationAgent.worst_case(
                    self.bellman_equation(v))),
            iterations=iterations)

    def state_action_value(self, iterations=100):
        return ValueIterationAgent.fixed_point_iteration(
            self.rewards,
            lambda q: RobustValueIterationAgent.worst_case(
                self.bellman_equation(
                    RobustValueIterationAgent.value_from_action_values(q))),
            iterations=iterations)

    @staticmethod
    def worst_case(model_action_values):
        return np.min(model_action_values, axis=0)

    def bellman_equation(self, value):
        v_shaped = value.reshape((np.shape(value)[0], 1, 1, np.shape(value)[1]))
        next_v = (self.transitions * v_shaped).sum(axis=-1)
        return self.rewards + self.config["gamma"] * next_v

    @staticmethod
    def fixed_point_iteration(initial, iterate, iterations):
        value = initial
        for _ in range(iterations):
            next_value = iterate(value)
            if np.allclose(value, next_value):
                break
            value = next_value
        return value

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
