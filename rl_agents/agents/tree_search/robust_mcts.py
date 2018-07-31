import numpy as np

from rl_agents.agents.abstract import AbstractAgent
from rl_agents.agents.common import load_agent


class DiscreteRobustMCTS(AbstractAgent):
    def __init__(self,
                 env,
                 config=None):
        """
            A new MCTS agent with multiple environment models.
        :param env: The true environment
        :param config: The agent configuration.
                       It should include an "agents" key giving a list of paths to several MCTSAgent configurations.
                       This class could be modified to directly load configuration dictionaries instead of agents
                       config files.
        """
        super(DiscreteRobustMCTS, self).__init__(config)
        self.agents = [load_agent(agent_config["path"], env) for agent_config in self.config["agents"]]
        self.__env = env

    @classmethod
    def default_config(cls):
        return dict(agents=[])

    @property
    def env(self):
        return self.__env

    @env.setter
    def env(self, env):
        self.__env = env
        for agent in self.agents:
            agent.env = env

    def plan(self, state):
        for agent in self.agents:
            agent.plan(state)

        min_action_values = {k: np.inf for k in self.env.ACTIONS.keys()}
        for agent in self.agents:
            min_action_values = {k: min(v, agent.mcts.root.children[k].value)
                                 for k, v in min_action_values.items()
                                 if k in agent.mcts.root.children}
        action = max(min_action_values.keys(), key=(lambda key: min_action_values[key]))
        for agent in self.agents:
            agent.previous_action = action

        return [action]

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def seed(self, seed=None):
        for agent in self.agents:
            seeds = agent.seed(seed)
            seed = seeds[0]
        return seed

    def record(self, state, action, reward, next_state, done):
        raise NotImplementedError()

    def act(self, state):
        return self.plan(state)[0]

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()


class IntervalRobustMCTS(AbstractAgent):
    def __init__(self, env, config=None):
        """
        :param highway_env.envs.abstract.AbstractEnv env: a highway-env environment
        :param config: the agent config
        """
        super(IntervalRobustMCTS, self).__init__(config)

        from highway_env.envs.abstract import AbstractEnv
        if not isinstance(env, AbstractEnv):
            raise ValueError("Only compatible with highway-env environments.")
        self.env = env
        self.robust_env = None
        self.reset()

    @classmethod
    def default_config(cls):
        return dict()

    def act(self, observation):
        self.robust_env = self.env.change_vehicles('highway_env.vehicle.uncertainty.IntervalVehicle')
        self.robust_env.step(1)
        self.robust_env.step(1)
        return 1

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()

    def record(self, state, action, reward, next_state, done):
        pass
