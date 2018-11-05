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
    plt.title('M')
    plt.subplot(312)
    plt.plot(budget, horizon)
    plt.title('L')
    plt.subplot(313)
    plt.plot(budget, episodes * K ** horizon)
    plt.axhline(y=K**(1/(1-gamma)), color='k')
    plt.title('Time / Memory complexity')
    plt.show()


def agent_configs():
    agents = {
        "olop": {
            "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
            "gamma": gamma,
            "max_depth": 2,
            "upper_bound": "hoeffding",
            "lazy_tree_construction": True
        },
        "kl-olop": {
            "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
            "gamma": gamma,
            "max_depth": 2,
            "upper_bound": "kullback-leibler",
            "lazy_tree_construction": True
        }
    }
    return agents


def evaluate(agent_config):
    environment_config = 'configs/FiniteMDPEnv/haystack/env3.json'
    gym.logger.set_level(gym.logger.INFO)
    env = load_environment(environment_config)
    agent = agent_factory(env, agent_config)
    evaluation = Evaluation(env,
                            agent,
                            directory=None,
                            num_episodes=1,
                            display_env=False,
                            display_agent=False,
                            display_rewards=False)
    evaluation.test()
    evaluation.close()
    return evaluation.monitor.stats_recorder.episode_rewards[0]


def main():
    n = np.arange(50, 1000, 25)
    M, L = allocate(n)

    agents = agent_configs()
    rewards = np.zeros((n.size, len(agents)))
    for i in range(n.size):
        print("Budget:", i+1, "/", n.size)
        for j in range(len(agents.keys())):
            config = list(agents.values())[j]
            config["budget"] = int(n[i])
            rewards[i, j] = evaluate(config)
    plt.plot(n, rewards)
    plt.legend(agents.keys())
    plot_budget(n, M, L)
    plt.show()


if __name__ == "__main__":
    main()
