"""Usage: planners_evaluation.py [options]

Compare performances of several planners

Options:
  -h --help
  --generate <true or false>  Generate new data [default: True].
  --show <true_or_false>      Plot results [default: True].
  --directory <path>          Specify directory path [default: ./out/planners].
  --data_file <path>          Specify output data file name [default: data.csv].
  --budgets <start,end,N>     Computational budgets available to planners, in logspace [default: 1,3,100].
  --seeds <(s,)n>             Number of evaluations of each configuration, with an optional first seed [default: 10].
  --processes <p>             Number of processes [default: 4]
  --chunksize <c>             Size of data chunks each processor receives [default: 1]
  --range <start:end>         Range of budgets to be plotted.
"""
from ast import literal_eval
from pathlib import Path

import tqdm
from docopt import docopt
from collections import OrderedDict
from itertools import product
from multiprocessing.pool import Pool

import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging

sns.set(font_scale=1.5, rc={'text.usetex': True})

from rl_agents.agents.common.factory import load_environment, agent_factory
from rl_agents.trainer.evaluation import Evaluation

logger = logging.getLogger(__name__)

gamma = 0.7
SEED_MAX = 1e9


def env_configs():
    # return ['configs/CartPoleEnv/env.json']
    # return ['configs/HighwayEnv/env_medium.json']
    # return ['configs/GridWorld/collect.json']
    return ['configs/FiniteMDPEnv/env_garnet.json']
    # return [Path("configs") / "DummyEnv" / "line_env.json"]


def agent_configs():
    agents = {
        "random": {
            "__class__": "<class 'rl_agents.agents.simple.random.RandomUniformAgent'>"
        },
        # "olop": {
        #     "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
        #     "gamma": gamma,
        #     "upper_bound": {
        #         "type": "hoeffding",
        #         "c": 4
        #     },
        #     "lazy_tree_construction": True,
        #     "continuation_type": "uniform",
        # },
        # "kl-olop": {
        #     "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
        #     "gamma": gamma,
        #     "upper_bound": {
        #         "type": "kullback-leibler",
        #         "threshold": "2*np.log(time) + 2*np.log(np.log(time))"
        #     },
        #     "lazy_tree_construction": True,
        #     "continuation_type": "uniform",
        # },
        "kl-olop": {
            "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
            "gamma": gamma,
            "upper_bound": {
                "type": "kullback-leibler",
                "threshold": "1*np.log(time)"
            },
            "lazy_tree_construction": True,
            "continuation_type": "uniform",
            # "env_preprocessors": [{"method": "simplify"}],
        },
        # "laplace": {
        #     "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
        #     "gamma": gamma,
        #     "upper_bound": {
        #         "type": "laplace",
        #         "c": 2
        #     },
        #     "lazy_tree_construction": True,
        #     "continuation_type": "uniform",
        # },
        # "opd": {
        #     "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
        #     "gamma": gamma,
        # },
        "mdp-gape": {
            "__class__": "<class 'rl_agents.agents.tree_search.mdp_gape.MDPGapEAgent'>",
            "gamma": gamma,
            "accuracy": 0.1,
            "confidence": 1,
            "upper_bound":
            {
                "type": "kullback-leibler",
                "time": "global",
                "threshold": "1*np.log(time)",
                "transition_threshold": "0.1*np.log(time)"
            },
            "max_next_states_count": 2,
            "continuation_type": "uniform",
            "step_strategy": "reset",
            # "env_preprocessors": [{"method": "simplify"}]
        },
        # "mdp-gape-conf": {
        #     "__class__": "<class 'rl_agents.agents.tree_search.mdp_gape.MDPGapEAgent'>",
        #     "gamma": gamma,
        #     "accuracy": 0.2,
        #     "confidence": 0.9,
        #     "upper_bound":
        #     {
        #         "type": "kullback-leibler",
        #         "time": "global",
        #         "threshold": "np.log(1/(1 - confidence)) + np.log(count)",
        #         "transition_threshold": "np.log(1/(1 - confidence)) + np.log(1 + np.log(count))"
        #     },
        #     "max_next_states_count": 2,
        #     "continuation_type": "uniform",
        #     "step_strategy": "reset",
        #     "horizon_from_accuracy": True,
        #     # "env_preprocessors": [{"method": "simplify"}]
        # },
        "brue": {
            "__class__": "<class 'rl_agents.agents.tree_search.brue.BRUEAgent'>",
            "gamma": gamma,
            "step_strategy": "reset",
        },
        "value_iteration": {
            "__class__": "<class 'rl_agents.agents.dynamic_programming.value_iteration.ValueIterationAgent'>",
            "gamma": gamma,
            "iterations": int(3 / (1 - gamma))
        }
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

    logger.debug("Evaluating agent {} with budget {} on seed {}".format(agent_name, budget, seed))

    # Compute true value
    compute_regret = True
    compute_return = False
    if compute_regret:
        value_iteration_agent = agent_factory(env, agent_configs()["value_iteration"])
        best_action = value_iteration_agent.act(env.mdp.state)
        action = agent.act(env.mdp.state)
        q = value_iteration_agent.state_action_value()
        simple_regret = q[env.mdp.state, best_action] - q[env.mdp.state, action]
        gap = q[env.mdp.state, best_action] - np.sort(q[env.mdp.state, :])[-2]

        # if hasattr(agent.planner, "budget_used"):
        #     budget = agent.planner.budget_used
    else:
        simple_regret = 0
        gap = 0

    if compute_return:
        # Evaluate
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
        cum_discount = lambda signal: np.sum([gamma**t * signal[t] for t in range(len(signal))])
        return_ = cum_discount(rewards)
        mean_return = np.mean([cum_discount(rewards[t:]) for t in range(len(rewards))])
    else:
        length = 0
        total_reward = 0
        return_ = 0
        mean_return = 0

    # Save results
    result = {
        "agent": agent_name,
        "budget": budget,
        "seed": seed,
        "total_reward": total_reward,
        "return": return_,
        "mean_return": mean_return,
        "length": length,
        "simple_regret": simple_regret,
        "gap": gap
    }

    df = pd.DataFrame.from_records([result])
    with open(path, 'a') as f:
        df.to_csv(f, sep=',', encoding='utf-8', header=f.tell() == 0, index=False)


def prepare_experiments(budgets, seeds, path):
    budgets = np.unique(np.logspace(*literal_eval(budgets)).astype(int))
    agents = agent_configs()
    for excluded_agent in ["value_iteration", "olop", "laplace"]:
        agents.pop(excluded_agent, None)

    seeds = seeds.split(",")
    first_seed = int(seeds[0]) if len(seeds) == 2 else np.random.randint(0, SEED_MAX, dtype=int)
    seeds_count = int(seeds[-1])
    seeds = (first_seed + np.arange(seeds_count)).tolist()
    envs = env_configs()
    paths = [path]
    experiments = list(product(seeds, budgets, agents.items(), envs, paths))
    return experiments


latex_names = {
    "simple_regret": "simple regret $r_n$",
    "total_reward": "total reward $R$",
    "mean_return": "mean return $E[R]$",
    "1/epsilon": r"${1}/{\epsilon}$",
    "mdp-gape-conf": r"\texttt{MDP-GapE}",
    "mdp-gape": r"\texttt{MDP-GapE}",
    "kl-olop": r"\texttt{KL-OLOP}",
    "brue": r"\texttt{BRUE}",
    "budget": r"budget $n$",
}


def rename_df(df):
    df = df.rename(columns=latex_names)
    for key, value in latex_names.items():
        df["agent"] = df["agent"].replace(key, value)
    return df


def rename(value, latex=True):
    return latex_names.get(value, value) if latex else value


def plot_all(data_file, directory, data_range):
    print("Reading data from {}".format(directory / data_file))
    df = pd.read_csv(str(directory / data_file))
    df = df[~df.agent.isin(['agent'])].apply(pd.to_numeric, errors='ignore')
    df = df.sort_values(by="agent")
    df = rename_df(df)
    if data_range:
        start, end = data_range.split(':')
        df = df[df["budget"].between(int(start), int(end))]
    print("Number of seeds found: {}".format(df.seed.nunique()))

    try:
        for field in ["total_reward", "return", "length", "mean_return", "simple_regret"]:
            fig, ax = plt.subplots()
            ax.set(xscale="log")
            if field in ["simple_regret"]:
                ax.set_yscale("symlog", linthreshy=1e-4)
            sns.lineplot(x=rename("budget"), y=rename(field), ax=ax, hue="agent", data=df)
            field_path = directory / "{}.pdf".format(field)
            fig.savefig(field_path, bbox_inches='tight')
            field_path = directory / "{}.png".format(field)
            fig.savefig(field_path, bbox_inches='tight')
            print("Saving {} plot to {}".format(field, field_path))
    except ValueError as e:
        print(e)

    custom_processing(df, directory)


def custom_processing(df, directory):
    return
    df = df[df["agent"] == "bai_mcts_conf"]
    print("Median values")
    print(df.median(axis=0))
    print("Maximum values")
    print(df.max(axis=0))
    for field in ["budget", "simple_regret"]:
        # histogram on linear scale
        _, bins, _ = plt.hist(df[field], bins=8)
        fig, ax = plt.subplots()
        if bins[0] > 0:
            logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            ax.set(xscale="log")
        elif bins[1] > 0:
            logbins = np.logspace(np.floor(np.log10(bins[1])), np.ceil(np.log10(bins[-1])), len(bins))
            ax.set_xscale("symlog", linthreshx=bins[1])
            logbins = np.insert(logbins, 0, 0)
        else:
            logbins = bins
        sns.distplot(df[field], bins=logbins, ax=ax, kde=False, rug=True)
        field_path = directory / "{}_hist.pdf".format(field)
        fig.savefig(field_path, bbox_inches='tight')
        field_path = directory / "{}_hist.png".format(field)
        fig.savefig(field_path, bbox_inches='tight')
        print("Saving {} plot to {}".format(field, field_path))


def main(args):
    if args["--generate"] == "True":
        experiments = prepare_experiments(args["--budgets"], args['--seeds'],
                                          str(Path(args["--directory"]) / args["--data_file"]))
        chunksize = int(args["--chunksize"])
        with Pool(processes=int(args["--processes"])) as p:
            list(tqdm.tqdm(p.imap_unordered(evaluate, experiments, chunksize=chunksize), total=len(experiments)))
    if args["--show"] == "True":
        plot_all(args["--data_file"], Path(args["--directory"]), args["--range"])


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
