"""Usage: planners_robust_evaluation.py [options]

Compare performances of several planners

Options:
  -h --help
  --generate <true or false>  Generate new data [default: True].
  --show <true_or_false>      Plot results [default: True].
  --filename <path>           Specify output data file path [default: data.csv].
  --directory <path>          Specify figure data file path [default: ./out/planners].
  --seeds <(s,)n>             Number of evaluations of each configuration, with an optional first seed [default: 10].
  --processes <p>             Number of processes [default: 4]
  --chunksize <c>             Size of data chunks each processor receives
  --range <start:end>         Range of budgets to be plotted.
"""
import itertools
from ast import literal_eval
from pathlib import Path

from docopt import docopt
from collections import OrderedDict
from itertools import product
from multiprocessing.pool import Pool

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rl_agents.agents.common.factory import load_environment, agent_factory, load_agent, safe_deepcopy_env
from rl_agents.trainer.evaluation import Evaluation
import os
os.environ['SDL_VIDEODRIVER'] = 'x11'

SEED_MAX = 1e9

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

sns.set_style("ticks")
# sns.set_palette("colorblind")


def env_configs():
    return [
        'configs/ObstacleEnv/env_obs_state.json',
        # 'configs/IntersectionEnv/env_multi_model.json'
    ]


def agent_configs():
    agents = {
        # "robust-epc": "configs/ObstacleEnv/robust-epc.json",
        "nominal-epc": "configs/ObstacleEnv/nominal.json",
        # "model-bias": "configs/ObstacleEnv/model_bias.json",
        # "oracle": "configs/ObstacleEnv/oracle.json",
        # "robust-epc": "configs/IntersectionEnv/agents/Robust/robust-epc.json",
        # "nominal-known-route": "configs/IntersectionEnv/agents/Robust/known_route.json",
        # "nominal-min-route": "configs/IntersectionEnv/agents/Robust/minimum_route.json",
        # "oracle": "configs/IntersectionEnv/agents/Robust/oracle.json",
    }
    return agents


def evaluate(experiment):
    # Prepare workspace
    seed, agent_config, env_config, path = experiment
    gym.logger.set_level(gym.logger.DISABLED)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Make environment
    env = load_environment(env_config)

    # Make agent
    agent_name, agent_config = agent_config
    agent = load_agent(agent_config, env)

    # Evaluate
    print("Evaluating agent {} on seed {}".format(agent_name, seed))
    evaluation = Evaluation(env,
                            agent,
                            directory=path.parent / agent_name,
                            num_episodes=1,
                            sim_seed=seed,
                            display_env=True,
                            display_agent=True,
                            display_rewards=False)
    estimate_value = False
    if estimate_value:
        rewards, values, terminal = [], [], False
        evaluation.seed(episode=0)
        evaluation.reset()
        evaluation.training = False
        gamma = 0.99 or agent.config["gamma"]
        while not terminal:
            # Estimate state value
            oracle_env = safe_deepcopy_env(agent.env)
            oracle = load_agent(agent_configs()["oracle"], oracle_env)
            oracle_done, oracle_rewards = False, []
            while not oracle_done:
                action = oracle.act(None)
                _, oracle_reward, oracle_done, _ = oracle_env.step(action)
                oracle_rewards.append(oracle_reward)
            value = np.sum([gamma**t * oracle_rewards[t] for t in range(len(oracle_rewards))])
            values.append(value)

            reward, terminal = evaluation.step()
            rewards.append(reward)
        evaluation.close()

        returns = [np.sum([gamma**t * rewards[k+t] for t in range(len(rewards[k:]))]) for k in range(len(rewards))]

        # Save intermediate results
        df = pd.DataFrame({
            "agent": agent_name,
            "time": range(len(rewards)),
            "seed": [seed] * len(rewards),
            "reward": rewards,
            "return": returns,
            "value": values
        })
    else:
        evaluation.test()
        rewards = evaluation.wrapped_env.episode_returns[0]
        length = evaluation.wrapped_env.episode_lengths[0]
        total_reward = np.sum(rewards)

        cum_discount = lambda signal, gamma: np.sum([gamma**t * signal[t] for t in range(len(signal))])
        return_ = cum_discount(rewards, 0.9)
        return_undisc = cum_discount(rewards, 0.99)
        result = {
            "agent": agent_name,
            "seed": seed,
            "total_reward": total_reward,
            "return": return_,
            "return_undisc": return_undisc,
            "length": length,
        }
        df = pd.DataFrame.from_records([result])
    with open(path, 'a') as f:
        df.to_csv(f, sep=',', encoding='utf-8', header=f.tell() == 0, index=False)


def prepare_experiments(seeds, path):
    agents = agent_configs()
    # agents = {a: v for a, v in agents.items() if a != "oracle"}
    seeds = seeds.split(",")
    if len(seeds) <= 2:
        first_seed = int(seeds[0]) if len(seeds) == 2 else np.random.randint(0, SEED_MAX, dtype=int)
        seeds_count = int(seeds[-1])
        seeds = (first_seed + np.arange(seeds_count)).tolist()
    else:
        seeds = [int(seed) for seed in seeds]
    envs = env_configs()
    paths = [path]
    experiments = list(product(seeds, agents.items(), envs, paths))
    return experiments


def plot_all(directory, filename, data_range):
    print("Reading data from {}".format(directory))
    df = pd.read_csv(directory / filename)
    if data_range:
        start, end = data_range.split(':')
        df = df[df["time"].between(int(start), int(end))]
    print("Number of seeds found: {}".format(df.seed.nunique()))
    try:
        custom_processing_return(df)
    except Exception:
        pass
    try:
        df["regret"] = (df["value"] - df["return"]).clip(lower=0)
        df[r"samples $N$"] = 5*df["time"]
        custom_processing_regret(df)
    except Exception:
        pass
    df = df.replace({
        "robust-epc": r"\texttt{Robust}",
        "nominal-epc": r"\texttt{Nominal}",
    })
    df = df.sort_values(by='agent')

    fig, ax = plt.subplots()
    ax.set_yscale("symlog", linthreshy=1e-4)
    sns.lineplot(x=r"samples $N$", y='regret', hue='agent', ax=ax, data=df)
    ax.set_ylabel("suboptimality")
    field = "regret"
    field_path = directory / "{}.pdf".format(field)
    fig.savefig(field_path, bbox_inches='tight')
    field_path = directory / "{}.png".format(field)
    fig.savefig(field_path, bbox_inches='tight')
    print("Saving {} plot to {}".format(field, field_path))

    df = df.groupby(["agent", "time"], as_index=False).max()
    df = df.rename(columns={"regret": "max regret"})

    palette = itertools.cycle(sns.color_palette())
    for agent in df["agent"].unique():
        color = next(palette)
        agent_df = df[df["agent"] == agent]
        ax.plot(agent_df["samples $N$"], agent_df["max regret"], color=color, linestyle="--")

    field = "max_regret"
    field_path = directory / "{}.pdf".format(field)
    fig.savefig(field_path, bbox_inches='tight')
    field_path = directory / "{}.png".format(field)
    fig.savefig(field_path, bbox_inches='tight')
    print("Saving {} plot to {}".format(field, field_path))


def custom_processing_regret(df):
    print("=== Duration ===")
    duration = df.groupby(["agent", "seed"]).max()
    print("Worst case")
    print(duration.groupby(["agent"]).min()["time"])
    print("Mean")
    print(duration.groupby(["agent"]).mean()["time"])
    print("Collisions")
    print(duration[duration["time"] < 19].groupby(["agent"]).count())

    print("=== Return ===")
    returns = df[df["time"] == 0]
    print("Worst case")
    print(returns.groupby(["agent"]).min())
    print("Mean")
    print(returns.groupby(["agent"]).mean())
    print("Std")
    print(returns.groupby(["agent"]).std())

def custom_processing_return(df):
    print("Worst case")
    print(df.groupby(["agent"]).min())
    print("Mean")
    print(df.groupby(["agent"]).mean())
    print("Std")
    print(df.groupby(["agent"]).std())
    print("Max")
    print(df.groupby(["agent"]).max())

    print("Successes")
    print(df[df["length"] == 14].groupby(["agent"]).count())
    print(df[(df["length"] < 14)])


def main(args):
    raise NotImplementedError('The Monitor wrapper which previously recorded statistics has been replaced by '
                              'RecordEpisodeStatistics, so this file needs to be updated.')
    if args["--generate"] == "True":
        experiments = prepare_experiments(args['--seeds'], Path(args["--directory"]) / args["--filename"])
        chunksize = int(args["--chunksize"]) if args["--chunksize"] else args["--chunksize"]
        with Pool(processes=int(args["--processes"])) as p:
            p.map(evaluate, experiments, chunksize=chunksize)
    if args["--show"] == "True":
        plot_all(Path(args["--directory"]), args["--filename"], args["--range"])


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
