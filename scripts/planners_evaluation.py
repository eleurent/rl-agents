"""Usage: planners_evaluation.py [options]

Compare performances of several planners

Options:
  -h --help
  --generate <true or false>  Generate new data [default: True].
  --show <true_or_false>      Plot results [default: True].
  --data_path <path>          Specify output data file path [default: ./out/planners/data.csv].
  --plot_path <path>          Specify figure data file path [default: ./out/planners].
  --budgets <start,end,N>     Computational budgets available to planners, in logspace [default: 1,3,100].
  --seeds <(s,)n>             Number of evaluations of each configuration, with an optional first seed [default: 10].
  --processes <p>             Number of processes [default: 4]
  --chunksize <c>             Size of data chunks each processor receives
  --range <start:end>         Range of budgets to be plotted.
"""
from ast import literal_eval
from pathlib import Path

from docopt import docopt
from collections import OrderedDict
from itertools import product
from multiprocessing.pool import Pool

import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rl_agents.agents.common.factory import load_environment, agent_factory
from rl_agents.trainer.evaluation import Evaluation

gamma = 0.8
SEED_MAX = 1e9


def env_configs():
    # return ['configs/CartPoleEnv/env.json']
    # return ['configs/HighwayEnv/env_medium.json']
    return ['configs/GridWorld/collect.json']


def agent_configs():
    agents = {
        "random": {
            "__class__": "<class 'rl_agents.agents.simple.random.RandomUniformAgent'>"
        },
        "olop": {
            "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
            "gamma": gamma,
            "max_depth": 4,
            "upper_bound": {
                "type": "hoeffding",
                "c": 4
            },
            "lazy_tree_construction": True,
            "continuation_type": "uniform",
            # "env_preprocessors": [{"method": "simplify"}]
        },
        "kl-olop": {
            "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
            "gamma": gamma,
            "max_depth": 4,
            "upper_bound": {
                "type": "kullback-leibler",
                "c": 2
            },
            "lazy_tree_construction": True,
            "continuation_type": "uniform",
            # "env_preprocessors": [{"method": "simplify"}]
        },
        "kl-olop-1": {
            "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
            "gamma": gamma,
            "max_depth": 4,
            "upper_bound": {
                "type": "kullback-leibler",
                "c": 1
            },
            "lazy_tree_construction": True,
            "continuation_type": "uniform",
            # "env_preprocessors": [{"method": "simplify"}]
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
            # "env_preprocessors": [{"method": "simplify"}]
        },
        "deterministic": {
            "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
            "gamma": gamma,
            # "env_preprocessors": [{"method": "simplify"}]
        }
        # ,
        # "value_iteration": {
        #     "__class__": "<class 'rl_agents.agents.dynamic_programming.value_iteration.ValueIterationAgent'>",
        #     "gamma": gamma,
        #     "iterations": int(3 / (1 - gamma))
        # }
    }
    return OrderedDict(agents)


def evaluate(experiment):
    # Prepare workspace
    seed, budget, agent_config, env_config, path = experiment
    gym.logger.set_level(gym.logger.DISABLED)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Make environment
    env = load_environment(env_config)

    # Make agent
    agent_name, agent_config = agent_config
    agent_config["budget"] = int(budget)
    agent = agent_factory(env, agent_config)

    # Evaluate
    print("Evaluating agent {} with budget {} on seed {}".format(agent_name, budget, seed))
    evaluation = Evaluation(env,
                            agent,
                            directory=Path("out") / "planners" / agent_name,
                            num_episodes=1,
                            sim_seed=seed,
                            display_env=False,
                            display_agent=False,
                            display_rewards=False)
    evaluation.test()
    rewards = evaluation.monitor.stats_recorder.episode_rewards_[0]
    length = evaluation.monitor.stats_recorder.episode_lengths[0]
    total_reward = np.sum(rewards)
    return_ = np.sum([gamma**t * rewards[t] for t in range(len(rewards))])

    # Save results
    result = {
        "agent": agent_name,
        "budget": budget,
        "seed": seed,
        "total_reward": total_reward,
        "return": return_,
        "length": length
    }
    df = pd.DataFrame.from_records([result])
    with open(path, 'a') as f:
        df.to_csv(f, sep=',', encoding='utf-8', header=f.tell() == 0, index=False)


def prepare_experiments(budgets, seeds, path):
    budgets = np.unique(np.logspace(*literal_eval(budgets)).astype(int))
    agents = agent_configs()

    seeds = seeds.split(",")
    first_seed = int(seeds[0]) if len(seeds) == 2 else np.random.randint(0, SEED_MAX, dtype=int)
    seeds_count = int(seeds[-1])
    seeds = (first_seed + np.arange(seeds_count)).tolist()
    envs = env_configs()
    paths = [path]
    experiments = list(product(seeds, budgets, agents.items(), envs, paths))
    return experiments


def plot_all(data_path, plot_path, data_range):
    print("Reading data from {}".format(data_path))
    df = pd.read_csv(data_path)
    df = df[~df.agent.isin(['agent'])].apply(pd.to_numeric, errors='ignore')
    df = df.sort_values(by="agent")
    if data_range:
        start, end = data_range.split(':')
        df = df[df["budget"].between(int(start), int(end))]
    print("Number of seeds found: {}".format(df.seed.nunique()))

    for field in ["total_reward", "return", "length"]:
        fig, ax = plt.subplots()
        ax.set(xscale="log")
        sns.lineplot(x="budget", y=field, ax=ax, hue="agent", data=df)
        field_path = plot_path / "{}.svg".format(field)
        fig.savefig(field_path, bbox_inches='tight')
        field_path = plot_path / "{}.png".format(field)
        fig.savefig(field_path, bbox_inches='tight')
        print("Saving {} plot to {}".format(field, field_path))


def main(args):
    if args["--generate"] == "True":
        experiments = prepare_experiments(args["--budgets"], args['--seeds'], args["--data_path"])
        chunksize = int(args["--chunksize"]) if args["--chunksize"] else args["--chunksize"]
        with Pool(processes=int(args["--processes"])) as p:
            p.map(evaluate, experiments, chunksize=chunksize)
    if args["--show"] == "True":
        plot_all(Path(args["--data_path"]), Path(args["--plot_path"]), args["--range"])


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


def plot_budget(budget, episodes, horizon, K=5):
    plt.figure()
    plt.subplot(311)
    plt.plot(budget, episodes, '+')
    plt.legend(["M"])
    plt.subplot(312)
    plt.plot(budget, horizon, '+')
    plt.legend(["L"])
    plt.subplot(313)
    plt.plot(budget, horizon / K ** (horizon - 1))
    plt.legend(['Computational complexity ratio'])
    plt.show()


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
