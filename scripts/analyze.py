"""
Usage:
  analyze run <run_folder>... [options]
  analyze benchmark <benchmark_file> [options]
  analyze -h | --help

Options:
  -h --help           Show this screen.
  --out <path>        Directory to save figures [default: .].
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
        for dir in run_directories:
            self.get_data(Path(dir))
        self.analyze()

    def get_data(self, base_directory):
        logging.info("Fetching data in {}".format(base_directory))
        runs = [self.get_run_dataframe(d) for d in base_directory.glob("*")]
        for run in runs:
            run["agent"] = rename(str(base_directory.name))
        logging.info("Found {} runs.".format(len(runs)))
        self.data = pd.concat([self.data] + runs)

    @staticmethod
    def get_run_dataframe(directory, gamma=0.95, subsample=10):
        run_data = MonitorV2.load_results(directory)

        data = {x: run_data[x] for x in [
            "episode_rewards",
            "episode_lengths",
        ]}
        df = pd.DataFrame(data)
        df["discounted_rewards"] = [np.sum([episode[t]*gamma**t for t in range(len(episode))])
                                    for episode in run_data["episode_rewards_"]]
        df["episode"] = np.arange(df.shape[0])
        df["total rewards"] = df["episode_rewards"].rolling(subsample).mean()
        df["total length"] = df["episode_lengths"].rolling(subsample).mean()
        df["collision"] = df["episode_lengths"] < df["episode_lengths"].max() - 1
        df["collision"] = df["collision"].rolling(subsample).mean()
        df = df.iloc[::subsample]
        return df

    def analyze(self):
        for field in ["total rewards", "total length", "collision"]:
            logging.info("Analyzing {}".format(field))
            fig, ax = plt.subplots()
            sns.lineplot(x="episode", y=field, hue="agent", data=self.data, ax=ax, ci=95)
            field_path = self.out / "{}.pdf".format(field)
            fig.savefig(field_path, bbox_inches='tight')
            plt.show()
            plt.close()


def rename(name):
    dictionary = {
        "ego_attention": "Ego-Attention",
        "mlp": "MLP"
    }
    return dictionary.get(name, name)


if __name__ == '__main__':
    main()


