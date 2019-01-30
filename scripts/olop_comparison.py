import os
from collections import OrderedDict
from itertools import product
from multiprocessing.pool import Pool

import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from rl_agents.agents.common import load_environment, agent_factory
from rl_agents.trainer.evaluation import Evaluation

gamma = 0.8
K = 5


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
        budget = np.array(budget).item()
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
    return OrderedDict(agents)


def evaluate(env_config, agent_config, budget):
    env = load_environment(env_config)
    name, agent_config = agent_config
    agent_config["budget"] = int(budget)
    agent_config["iterations"] = int(env.config["max_steps"])

    print("Evaluating {} with budget {}".format(name, budget))
    gym.logger.set_level(gym.logger.DISABLED)
    agent = agent_factory(env, agent_config)
    evaluation = Evaluation(env,
                            agent,
                            directory=os.path.join(Evaluation.OUTPUT_FOLDER,
                                                   "olop_comparison",
                                                   "{}_{}".format(name, budget)),
                            training=False,
                            num_episodes=1,
                            display_env=False,
                            display_agent=False,
                            display_rewards=False)
    evaluation.monitor.stats_recorder.gamma = gamma
    evaluation.test()
    evaluation.close()

    rewards = np.array(evaluation.monitor.stats_recorder.episode_rewards_)
    value = np.sum(rewards * gamma ** np.arange(rewards.size))
    return value


def main():
    n = np.arange(50, 1000, 50)
    M, L = allocate(n)

    envs = ['configs/FiniteMDPEnv/env_garnet.json']
    agents = agent_configs()
    experiments = list(product(envs, agents.items(), n))

    with Pool(processes=4) as pool:
        results = pool.starmap(evaluate, experiments)

    values = pd.DataFrame()
    for experiment, value in zip(experiments, results):
        _, agent, budget = experiment
        values.at[budget, agent[0]] = value

    # plt.figure()
    values.plot()
    # plot_budget(n, M, L)
    plt.show()


if __name__ == "__main__":
    main()
