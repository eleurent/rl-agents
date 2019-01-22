import gym
import numpy as np
import matplotlib.pyplot as plt

from rl_agents.agents.common import load_environment, agent_factory
from rl_agents.trainer.evaluation import Evaluation

gamma = 0.8
K = 3


def olop_horizon(episodes, gamma):
    return int(np.ceil(np.log(episodes) / (2 * np.log(1 / gamma))))


def allocate(budget):
    if np.size(budget) > 1:
        episodes = np.zeros(budget.shape)
        horizon = np.zeros(budget.shape)
        for i in range(budget.size):
            episodes[i], horizon[i] = allocate(budget[i])
        return episodes, horizon
    else:
        for episodes in range(1, budget):
            if episodes * olop_horizon(episodes, gamma) > budget:
                episodes -= 1
                break
        horizon = olop_horizon(episodes, gamma)
        return episodes, horizon


def plot_budget(budget, episodes, horizon):
    plt.figure()
    plt.subplot(311)
    plt.plot(budget, episodes)
    plt.legend(["M"])
    plt.subplot(312)
    plt.plot(budget, horizon)
    plt.legend(["L"])
    plt.subplot(313)
    plt.plot(budget, horizon / K ** (horizon - 1))
    plt.legend(['Computational complexity ratio'])
    plt.show()


def agent_configs():
    agents = {
        "value-iteration": {
            "__class__": "<class 'rl_agents.agents.dynamic_programming.value_iteration.ValueIterationAgent'>",
            "gamma": gamma,
            "iterations": 2/(1-gamma)
        },
        "olop": {
            "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
            "gamma": gamma,
            "max_depth": 2,
            "upper_bound": {"type": "hoeffding"},
            "lazy_tree_construction": True
        },
        "kl-olop": {
            "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
            "gamma": gamma,
            "max_depth": 2,
            "upper_bound": {"type": "kullback-leibler"},
            "lazy_tree_construction": True
        }
        # ,
        # "kl-olop-m": {
        #     "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
        #     "gamma": gamma,
        #     "max_depth": 2,
        #     "upper_bound": {"type": "hoeffding", "time": "local"},
        #     "lazy_tree_construction": True
        #  }
    }
    return agents


def evaluate(agent_config, env):
    gym.logger.set_level(gym.logger.DISABLED)
    agent = agent_factory(env, agent_config)
    evaluation = Evaluation(env,
                            agent,
                            directory=None,
                            num_episodes=1,
                            display_env=False,
                            display_agent=False,
                            display_rewards=False)
    evaluation.monitor.stats_recorder.gamma = gamma
    evaluation.test()
    evaluation.close()
    return evaluation.monitor.stats_recorder.episode_values[0]


def main():
    n = np.arange(50, 500, 100)
    M, L = allocate(n)

    environment_config = 'configs/FiniteMDPEnv/env_garnet.json'
    agents = agent_configs()
    returns = np.zeros((n.size, len(agents)))
    for i in range(n.size):
        # Environment creation
        env = load_environment(environment_config)
        print("Budget:", i+1, "/", n.size)
        for j in range(len(agents.keys())):
            env.reset()
            config = list(agents.values())[j]
            config["budget"] = int(n[i])
            config["iterations"] = int(env.config["max_steps"])
            returns[i, j] += evaluate(config, env)
    plt.figure()
    plt.plot(n, returns)
    plt.legend(agents.keys())
    plot_budget(n, M, L)
    plt.show()


if __name__ == "__main__":
    main()
