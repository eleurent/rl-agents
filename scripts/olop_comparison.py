import os
from collections import OrderedDict
from itertools import product
from multiprocessing.pool import Pool

import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rl_agents.agents.common import load_environment, agent_factory
from rl_agents.trainer.evaluation import Evaluation

gamma = 0.8
K = 5
DATA_PATH = os.path.join(Evaluation.OUTPUT_FOLDER, "olop_data.xlsx")
FIGURE_PATH = os.path.join(Evaluation.OUTPUT_FOLDER, "olop_plot.png")


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
    }
    return OrderedDict(agents)


def value_iteration():
    return {
        "__class__": "<class 'rl_agents.agents.dynamic_programming.value_iteration.ValueIterationAgent'>",
        "gamma": gamma,
        "iterations": int(3 / (1 - gamma))
    }


def evaluate(env_config, agent_config, budget, seed):
    gym.logger.set_level(gym.logger.DISABLED)
    env = load_environment(env_config)
    env.configure({"seed": seed})
    env.seed(seed)
    state = env.reset()

    name, agent_config = agent_config
    print("Evaluating {} with budget {}".format(name, budget))
    agent_config["budget"] = int(budget)
    agent_config["iterations"] = int(env.config["max_steps"])
    agent = agent_factory(env, agent_config)
    agent.seed(seed)
    action = agent.act(state)

    values = agent_factory(env, value_iteration()).state_action_value()[env.mdp.state, :]
    return values[action],  np.amax(values)


def append_df_to_excel(filename, df, sheet_name='Sheet1', startrow=None,
                       truncate_sheet=False,
                       **to_excel_kwargs):
    """
    Append a DataFrame [df] to existing Excel file [filename]
    into [sheet_name] Sheet.
    If [filename] doesn't exist, then this function will create it.

    Parameters:
      filename : File path or existing ExcelWriter
                 (Example: '/path/to/file.xlsx')
      df : dataframe to save to workbook
      sheet_name : Name of sheet which will contain DataFrame.
                   (default: 'Sheet1')
      startrow : upper left cell row to dump data frame.
                 Per default (startrow=None) calculate the last row
                 in the existing DF and write to the next row...
      truncate_sheet : truncate (remove and recreate) [sheet_name]
                       before writing DataFrame to Excel file
      to_excel_kwargs : arguments which will be passed to `DataFrame.to_excel()`
                        [can be dictionary]

    Returns: None
    """
    from openpyxl import load_workbook

    import pandas as pd

    # ignore [engine] parameter if it was passed
    if 'engine' in to_excel_kwargs:
        to_excel_kwargs.pop('engine')

    writer = pd.ExcelWriter(filename, engine='openpyxl')

    try:
        # try to open an existing workbook
        writer.book = load_workbook(filename)

        # get the last row in the existing Excel sheet
        # if it was not specified explicitly
        if startrow is None and sheet_name in writer.book.sheetnames:
            startrow = writer.book[sheet_name].max_row

        # truncate sheet
        if truncate_sheet and sheet_name in writer.book.sheetnames:
            # index of [sheet_name] sheet
            idx = writer.book.sheetnames.index(sheet_name)
            # remove [sheet_name]
            writer.book.remove(writer.book.worksheets[idx])
            # create an empty sheet [sheet_name] using old index
            writer.book.create_sheet(sheet_name, idx)

        # copy existing sheets
        writer.sheets = {ws.title: ws for ws in writer.book.worksheets}
        to_excel_kwargs["header"] = None
    except FileNotFoundError:
        # file does not exist yet, we will create it
        pass

    if startrow is None:
        startrow = 0

    # write out the new sheet
    df.to_excel(writer, sheet_name, startrow=startrow, **to_excel_kwargs)
    # save the workbook
    writer.save()
    print("Saving dataframe to {}".format(filename))


def store_results(experiments, results):
    print("Finished! Exporting dataframe...")
    df = pd.DataFrame(columns=['agent', 'budget', 'value', 'optimal_value', 'seed'])
    for experiment, result in zip(experiments, results):
        _, agent, budget, seed = experiment
        value, optimal_value = result
        df = df.append({"agent": agent[0],
                        "budget": budget,
                        "seed": seed,
                        "value": value,
                        "optimal_value": optimal_value},
                       ignore_index=True)
    df["regret"] = df["optimal_value"] - df["value"]
    append_df_to_excel(DATA_PATH, df, index=False)
    return df


def plot_all(df=None):
    if not df:
        print("Loading dataframe from {}".format(DATA_PATH))
        df = pd.read_excel(DATA_PATH)
    fig, ax = plt.subplots()
    sns.lineplot(x="budget", y="regret", ax=ax, hue="agent", data=df, markers=True)
    plt.savefig(FIGURE_PATH)
    # plt.show()


def main():
    n = np.arange(5, 50, 5)**2
    M, L = allocate(n)

    envs = ['configs/FiniteMDPEnv/env_garnet.json']
    agents = agent_configs()
    seeds = np.random.randint(0, 1000, 5, dtype=int).tolist()
    experiments = list(product(envs, agents.items(), n, seeds))

    with Pool(processes=4) as pool:
        results = pool.starmap(evaluate, experiments)
    store_results(experiments, results)
    plot_all()


if __name__ == "__main__":
    main()
