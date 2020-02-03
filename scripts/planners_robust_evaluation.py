"""Usage: planners_robust_evaluation.py [options]

Compare performances of several planners

Options:
  -h --help
  --generate <true or false>  Generate new data [default: True].
  --show <true_or_false>      Plot results [default: True].
  --data_path <path>          Specify output data file path [default: ./out/planners/data.csv].
  --plot_path <path>          Specify figure data file path [default: ./out/planners].
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

from rl_agents.agents.common.factory import load_environment, agent_factory, load_agent, safe_deepcopy_env
from rl_agents.trainer.evaluation import Evaluation

SEED_MAX = 1e9


def env_configs():
    return ['configs/ObstacleEnv/env_obs_state.json']


def agent_configs():
    agents = {
        "robust-epc": "configs/ObstacleEnv/RobustEPC.json",
        "nominal-epc": "configs/ObstacleEnv/NominalEPC.json",
        # "model-bias": "configs/ObstacleEnv/ModelBias.json",
        "oracle": "configs/ObstacleEnv/DeterministicPlannerAgent.json",
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
                            directory=Path("out") / "planners" / agent_name,
                            num_episodes=1,
                            sim_seed=seed,
                            display_env=False,
                            display_agent=False,
                            display_rewards=False)
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
    with open(path, 'a') as f:
        df.to_csv(f, sep=',', encoding='utf-8', header=f.tell() == 0, index=False)


def prepare_experiments(seeds, path):
    agents = agent_configs()
    agents = {a: v for a, v in agents.items() if a != "oracle"}
    seeds = seeds.split(",")
    first_seed = int(seeds[0]) if len(seeds) == 2 else np.random.randint(0, SEED_MAX, dtype=int)
    seeds_count = int(seeds[-1])
    seeds = (first_seed + np.arange(seeds_count)).tolist()
    envs = env_configs()
    paths = [path]
    experiments = list(product(seeds, agents.items(), envs, paths))
    return experiments


def plot_all(data_path, plot_path, data_range):
    print("Reading data from {}".format(data_path))
    df = pd.read_csv(data_path)
    if data_range:
        start, end = data_range.split(':')
        df = df[df["time"].between(int(start), int(end))]
    print("Number of seeds found: {}".format(df.seed.nunique()))
    df["regret"] = df["value"] - df["return"]

    fig, ax = plt.subplots()
    sns.lineplot(x="time", y='regret', hue='agent', ax=ax, data=df)
    field = "regret"
    field_path = plot_path / "{}.pdf".format(field)
    fig.savefig(field_path, bbox_inches='tight')
    field_path = plot_path / "{}.png".format(field)
    fig.savefig(field_path, bbox_inches='tight')
    print("Saving {} plot to {}".format(field, field_path))


def main(args):
    if args["--generate"] == "True":
        experiments = prepare_experiments(args['--seeds'], args["--data_path"])
        chunksize = int(args["--chunksize"]) if args["--chunksize"] else args["--chunksize"]
        # for e in experiments:
        #     evaluate(e)
        with Pool(processes=int(args["--processes"])) as p:
            p.map(evaluate, experiments, chunksize=chunksize)
    if args["--show"] == "True":
        plot_all(Path(args["--data_path"]), Path(args["--plot_path"]), args["--range"])


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
