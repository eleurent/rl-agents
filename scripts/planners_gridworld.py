import copy
from collections import defaultdict

import gym
import highway_env
from pathlib import Path
import itertools
import numpy as np
from gym import spaces, Env
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import matplotlib

# matplotlib.use('Qt5Agg')

from rl_agents.agents.tree_search.graphics import TreePlot
from rl_agents.agents.common.factory import agent_factory, load_environment

sns.set()
out = Path("out/planners")


class GridEnv(Env):
    def __init__(self):
        dt = 0.1
        self.x = np.zeros((2))
        self.action_space = spaces.Discrete(8)

    def step(self, action):
        if action == 0:
            self.x[0] += 1
        elif action == 1:
            self.x[1] += 1
        elif action == 2:
            self.x[0] -= 1
        elif action == 3:
            self.x[1] -= 1
        elif action == 4:
            self.x[0] += 1
            self.x[1] += 1
        elif action == 5:
            self.x[0] += 1
            self.x[1] -= 1
        elif action == 6:
            self.x[0] -= 1
            self.x[1] += 1
        elif action == 7:
            self.x[0] -= 1
            self.x[1] -= 1
        return self.x, self.reward(), False, {}

    def reward(self):
        return np.clip((self.x[0] > 3 and self.x[1] > 3) * ((self.x[0] + self.x[1] - 6) / 12), 0, 1)

    def reset(self):
        self.x = np.array([0, 0])
        return self.x

    def simplify(self):
        return self

    def seed(self, seed):
        pass

gamma = 0.9

agents = {
    "deterministic": {
        "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
        "gamma": gamma,
        "env_preprocessors": [{"method": "simplify"}]
    },
    "state_aware": {
        "__class__": "<class 'rl_agents.agents.tree_search.state_aware.StateAwarePlannerAgent'>",
        "gamma": gamma,
        "env_preprocessors": [{"method": "simplify"}]
    },
}


def get_trajs(node, state, obs=None):
    trajs = []
    if obs is None:
        obs = state.reset()
    if node.children:
        for action, child in node.children.items():
            next_state = copy.deepcopy(state)
            next_obs, _, _, _ = next_state.step(action)
            child_trajs = get_trajs(child, next_state, next_obs)
            trajs.extend([[obs.tolist()] + t for t in child_trajs])
    else:
        trajs = [[obs.tolist()]]
    return trajs


def evaluate(env, agent_name, budget=8*(8**4 - 1)/(8 - 1), seed=None):
    print("Evaluating", agent_name)
    agent_config = agents[agent_name]
    agent_config["budget"] = budget
    agent = agent_factory(env, agent_config)
    if seed is not None:
        agent.seed(seed)
        env.seed(seed)
        obs = env.reset()
    agent.act(obs)
    return agent


def evaluate_agents(env, seed):
    for agent_name in agents.keys():
        env.seed(seed)
        env.reset()
        agent = evaluate(env, agent_name, seed=seed)
        yield agent, agent_name


def compare_agents(env, seed=0, show_tree=False, show_trajs=False, show_states=True):
    trajs = {}
    states_freqs = {}
    for agent, agent_name in evaluate_agents(env, seed):
        if show_tree:
            TreePlot(agent.planner, max_depth=100).plot(out / "{}.svg".format(agent_name), title=agent_name)
            plt.show()
        if show_trajs or show_states:
            trajs[agent_name] = get_trajs(agent.planner.root, env)
            # Aggregate visits
            visits = defaultdict(int)
            for traj in trajs[agent_name]:
                for s in traj:
                    visits[str(s)] += 1
            lims = 10
            states_freqs[agent_name] = np.zeros((2 * lims + 1, 2 * lims + 1))
            for i, x in enumerate(np.arange(-lims, lims)):
                for j, y in enumerate(np.arange(-lims, lims)):
                    states_freqs[agent_name][i, j] = visits[str([x, y])]

    if show_states:
        vmax = max([st.max() for st in states_freqs.values()])
        for agent_name, states in states_freqs.items():
            cmap = plt.cm.coolwarm
            fig, ax = plt.subplots()
            img = ax.imshow(states.T,
                            extent=(-lims, lims, -lims, lims),
                            norm=colors.LogNorm(vmax=vmax),
                            cmap=cmap)
            fig.colorbar(img, ax=ax)
            plt.title(agent_name)
            plt.savefig(out / "states_{}.svg".format(agent_name))
            plt.show()

    if show_trajs:
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

    env = GridEnv()
    # env = gym.make("highway-v0")
    compare_agents(env, show_tree=True)
