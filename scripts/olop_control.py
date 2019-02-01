import copy

import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

from rl_agents.agents.common import agent_factory


class DynamicsEnv:
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
        self.x = np.random.rand((2, 1))


gamma = 0.9
agents = {
    "olop": {
        "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
        "gamma": gamma,
        "upper_bound": {"type": "hoeffding"},
        "lazy_tree_construction": True
    },
    "kl-olop": {
        "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
        "gamma": gamma,
        "upper_bound": {"type": "kullback-leibler"},
        "lazy_tree_construction": True
    },
    "deterministic": {
        "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
        "gamma": gamma
    }
}


def get_trajs(node, env):
    trajs = []
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


def evaluate():
    env = DynamicsEnv()
    env.x = np.array([[-1], [0]])

    agent_config = agents["kl-olop"]
    agent_config["budget"] = 20000
    agent = agent_factory(env, agent_config)
    a = agent.act(env.x)
    print(agent.planner.root.children[0].count, agent.planner.root.children[1].count)
    trajs = get_trajs(agent.planner.root, env)
    for traj in trajs:
        x, y = zip(*traj)
        plt.plot(x, y, color='k', linestyle='dotted', linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    evaluate()
