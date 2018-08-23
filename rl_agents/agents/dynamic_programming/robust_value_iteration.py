import numpy as np

from rl_agents.agents.dynamic_programming.value_iteration import ValueIterationAgent


class RobustValueIterationAgent(ValueIterationAgent):
    def __init__(self, env, config=None):
        super(ValueIterationAgent, self).__init__(config)
        self.env = env
        self.mode = None
        self.transitions = np.array([])  # Dimension: M x S x A (x S)
        self.rewards = np.array([])  # Dimension: M x S x A
        self.models_from_config()

    @classmethod
    def default_config(cls):
        return dict(gamma=1.0,
                    models=[])

    def models_from_config(self):
        if not self.config.get("models", None):
            raise ValueError("No finite MDP model provided in agent configuration")

        self.mode = self.config["models"][0]["mode"]  # Assume all modes are the same
        self.transitions = np.array([mdp["transition"] for mdp in self.config["models"]])
        self.rewards = np.array([mdp["reward"] for mdp in self.config["models"]])

    def act(self, state):
        return np.argmax(self.state_action_value()[state, :])

    def state_value(self, iterations=100):
        return ValueIterationAgent.fixed_point_iteration(
            np.zeros((self.env.observation_space.n,)),
            lambda v: RobustValueIterationAgent.best_action_value(
                RobustValueIterationAgent.worst_case(
                    self.bellman_expectation(v))),
            iterations=iterations)

    def state_action_value(self, iterations=100):
        return ValueIterationAgent.fixed_point_iteration(
            np.zeros(np.shape(self.transitions)[1:3]),
            lambda q: RobustValueIterationAgent.worst_case(
                self.bellman_expectation(
                    RobustValueIterationAgent.best_action_value(q))),
            iterations=iterations)

    @staticmethod
    def worst_case(model_action_values):
        return np.min(model_action_values, axis=0)

    def bellman_expectation(self, value):
        if self.mode == "deterministic":
            next_v = value[self.transitions]
        elif self.mode == "stochastic":
            v_shaped = value.reshape((1, 1, 1, np.size(value)))
            next_v = (self.transitions * v_shaped).sum(axis=-1)
        else:
            raise ValueError("Unknown mode")
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
