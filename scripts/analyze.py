"""
Usage:
  analyze run <run_folder>... [options]
  analyze benchmark <benchmark_file> [options]
  analyze -h | --help

Options:
  -h --help           Show this screen.
  --out <path>        Directory to save figures [default: out].
  --first <episodes>  Use only the N first episodes of the runs.
  --last <episodes>   Use only the N last episodes of the runs.
"""
import json
from docopt import docopt
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from rl_agents.trainer.monitor import MonitorV2

logging.basicConfig(level=logging.INFO)
sns.set()


def main():
    opts = docopt(__doc__)
    episodes_range = [None, None]
    if opts['--first']:
        episodes_range[1] = int(opts['--first'])
    if opts['--last']:
        episodes_range[0] = -int(opts['--last'])
    if opts['run']:
        RunAnalyzer(opts['<run_folder>'], out=opts["--out"], episodes_range=episodes_range)
    elif opts['benchmark']:
        with open(opts['<benchmark_file>'], 'r') as f:
            RunAnalyzer(json.loads(f.read()), out=opts["--out"], episodes_range=episodes_range)


class RunAnalyzer(object):
    def __init__(self, run_directories, out, episodes_range=None):
        self.data = pd.DataFrame()
        self.out = Path(out)
        self.episodes_range = episodes_range
        for dir in run_directories:
            self.get_data(Path(dir))
        self.analyze()

    def get_data(self, base_directory):
        logging.info("Fetching data in {}".format(base_directory))
        agent_name = rename(str(base_directory.name))
        runs = [self.get_run_dataframe(d, agent_name=agent_name) for d in base_directory.glob("*")]
        logging.info("Found {} runs.".format(len(runs)))
        self.data = pd.concat([self.data] + runs)

    def get_run_dataframe(self, directory, agent_name='', gamma=0.95, subsample=10):
        run_data = MonitorV2.load_results(directory)
        if not run_data:
            return pd.DataFrame()

        # Common fields
        data = {
            "episode": np.arange(np.size(run_data["episode_rewards"])),
            "total reward": run_data["episode_rewards"],
            "discounted rewards": [np.sum([episode[t] * gamma ** t for t in range(len(episode))])
                                   for episode in run_data["episode_rewards_"]],
            "length": run_data["episode_lengths"],
        }

        # Additional highway-env fields
        try:
            dt = 1.0
            data.update({
                "crashed": [np.any(episode) for episode in run_data["episode_crashed"]],
                "speed": [np.mean(episode) for episode in run_data["episode_speed"]],
                "distance": [np.sum(episode)*dt for episode in run_data["episode_distance"]],
            })
        except KeyError as e:
            print(e)

        # Tags
        df = pd.DataFrame(data)
        df["run"] = str(directory.name)
        df["agent"] = agent_name

        # Filtering
        for field in ["total reward", "discounted rewards", "length", "crashed", "speed", "distance"]:
            try:
                df[field] = df[field].rolling(subsample).mean()
            except KeyError:
                continue

        # Subsample
        df = df.iloc[self.episodes_range[0]:self.episodes_range[1]:subsample]
        return df

    def analyze(self):
        self.find_best_run()
        for field in ["total reward", "length", "crashed", "speed", "distance"]:
            logging.info("Analyzing {}".format(field))
            fig, ax = plt.subplots()
            sns.lineplot(x="episode", y=field, hue="agent", data=self.data, ax=ax, ci=95)
            field_path = self.out / "{}.pdf".format(field)
            fig.savefig(field_path, bbox_inches='tight')
            plt.show()
            plt.close()

    def find_best_run(self, criteria="total reward", ascending=False):
        """ Maximal final total rewards"""
        df = self.data
        try:
            df = df[df["episode"] == df["episode"].max()].sort_values(criteria, ascending=ascending)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("Run with highest {}:".format(criteria))
                print(df.iloc[0])
        except IndexError:
            print("Could not find run matching desired criteria.")
        return self.data[self.data["run"] == df.iloc[0]["run"]]


def rename(name):
    dictionary = {
        "ego_attention": "Ego-Attention",
        "mlp": "FCN/List",
        "grid": "FCN/Grid",
        "grid_convnet2": "CNN/Grid"
    }
    return dictionary.get(name, name)


if __name__ == '__main__':
    main()


