from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rl_agents.trainer.monitor import MonitorV2

sns.set()


class RunAnalyzer(object):
    WINDOW = 20

    def __init__(self, run_directories, episodes_range=None):
        self.data = pd.DataFrame()
        for dir in run_directories:
            self.get_data(dir)
        self.analyze()

    def get_data(self, base):
        subdirectories = Path(base).glob("*")
        subdir_dfs = [self.get_run_dataframe(d) for d in subdirectories]
        for df in subdir_dfs:
            df["base"] = str(base)
        self.data = pd.concat([self.data] + subdir_dfs)

    @staticmethod
    def get_run_dataframe(directory, gamma=0.95):
        data = MonitorV2.load_results(directory)
        data["discounted_rewards"] = [np.sum([episode[t]*gamma**t for t in range(len(episode))])
                                      for episode in data["episode_rewards_"]]
        data["episode"] = np.arange(len(data["episode_lengths"]))
        data = {x: data[x] for x in [
            "episode",
            "episode_rewards",
            "episode_lengths",
            "discounted_rewards",
        ]}
        df = pd.DataFrame(data)
        return df

    def analyze(self):
        sns.regplot(x="episode", y="episode_rewards", hue="base", data=self.data, ci=95, n_boot=500)
        sns.regplot(x=x, y=y, )
        # sns.lineplot(x="episode", y="episode_rewards", hue="base", data=self.data, ci=95, n_boot=500)
        plt.show()
        plt.close()
