import copy

import gym
import highway_env
from pathlib import Path
import itertools
import numpy as np
from gym import spaces, Env
import matplotlib.pyplot as plt
import seaborn as sns

from rl_agents.agents.tree_search.graphics import TreePlot
from rl_agents.agents.common import agent_factory, load_environment

sns.set()
out = Path("out/planners")


class DynamicsEnv(Env):
    def __init__(self):
        dt = 0.1
        self.x = np.random.random((2, 1))
        self.A = np.array([[1, dt], [0, 1]])
        self.B = np.array([[0], [dt]])
        self.action_space = spaces.Discrete(2)

    def step(self, action):
        u = np.array([[2*action - 1]])
        self.x = self.A @ self.x + self.B @ u
        return self.x, self.reward(), False, {}

    def reward(self):
        return max(1 - self.x[0, 0]**2, 0)

    def reset(self):
        # self.x = np.random.random((2, 1))
        self.x = np.array([[-1], [0]])

    def simplify(self):
        return self

    def seed(self, seed):
        pass


env_zero_one = {
    "id": "finite-mdp-v0",
    "import_module": "finite_mdp",
    "mode": "deterministic",
    "transition": [[0, 0]],
    "reward": [[0, 0.7]],
    "terminal": [0,
                 0]
}

gamma = 0.9
agents = {
    "olop": {
        "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
        "gamma": gamma,
        "upper_bound": {
            "type": "hoeffding",
            "c": 4
        },
        "lazy_tree_construction": True,
        "continuation_type": "uniform",
        "env_preprocessors": [{"method": "simplify"}]
    },
    "kl-olop": {
        "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
        "gamma": gamma,
        "upper_bound": {
            "type": "kullback-leibler",
            "c": 2
        },
        "lazy_tree_construction": True,
        "continuation_type": "uniform",
        "env_preprocessors": [{"method": "simplify"}]
    },
    "kl-olop-1": {
        "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
        "gamma": gamma,
        "upper_bound": {
            "type": "kullback-leibler",
            "c": 1
        },
        "lazy_tree_construction": True,
        "continuation_type": "uniform",
        "env_preprocessors": [{"method": "simplify"}]
    },
    "laplace": {
        "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
        "gamma": gamma,
        "upper_bound": {
            "type": "laplace",
            "c": 2
        },
        "lazy_tree_construction": True,
        "continuation_type": "uniform",
        "env_preprocessors": [{"method": "simplify"}]
    },
    "deterministic": {
        "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
        "gamma": gamma,
        "env_preprocessors": [{"method": "simplify"}]
    }
}


def get_trajs(node, env):
    trajs = []
    if not isinstance(env, DynamicsEnv):
        return trajs
    if node.children:
        for action, child in node.children.items():
            state = copy.deepcopy(env)
            x, _, _, _ = state.step(action)
            child_trajs = get_trajs(child, state)
            if child_trajs:
                trajs.extend([[x.tolist()] + t for t in child_trajs])
            else:
                trajs.append([x.tolist()])
    return trajs


def evaluate(env, agent_name, budget=2000, seed=None):
    print("Evaluating", agent_name)
    agent_config = agents[agent_name]
    agent_config["budget"] = budget
    agent = agent_factory(env, agent_config)
    if seed is not None:
        agent.seed(seed)
    agent.act(env)
    return agent


def compare_trees(env, seed=0):
    for agent_name in agents.keys():
        env.seed(seed)
        env.reset()
        agent = evaluate(env, agent_name, seed=seed)
        TreePlot(agent.planner, max_depth=100).plot(out / "{}.svg".format(agent_name), title=agent_name)
        plt.show()


def compare_trajs(env, seed=0):
    trajs = {}
    for agent_name in agents.keys():
        env.seed(seed)
        env.reset()
        agent = evaluate(env, agent_name, seed=seed)
        trajs[agent_name] = get_trajs(agent.planner.root, env)

    palette = itertools.cycle(sns.color_palette())
    for agent, agent_trajs in trajs.items():
        color = next(palette)
        for traj in agent_trajs:
            x, y = zip(*traj)
            plt.plot(x, y, color=color, linestyle='dotted', linewidth=0.5)
    plt.savefig(out / "trajectories.png")
    plt.show()


if __name__ == "__main__":
    gym.logger.set_level(gym.logger.DEBUG)

    # env = DynamicsEnv()
    # env = gym.make("highway-v0")
    env = load_environment(env_zero_one)
    compare_trees(env, seed=5)
