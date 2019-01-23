import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from rl_agents.trainer.monitor import MonitorV2

sns.set()


class RunAnalyzer(object):
    WINDOW = 20

    def __init__(self, run_directories, episodes_range=None):
        self.base = os.path.commonprefix(run_directories) if len(run_directories) > 1 else ''
        self.episodes_range = episodes_range or [None, None]
        self.analyze(run_directories)

    def suffix(self, directory):
        return directory[len(self.base):]

    def analyze(self, run_directories):
        runs = {self.suffix(directory): MonitorV2.load_results(directory) for directory in run_directories}
        runs = {key: value for (key, value) in runs.items() if value is not None}
        self.plot_all(runs, field='episode_rewards', title='rewards')
        self.histogram_all(runs, field='episode_rewards', title='rewards')
        self.describe_all(runs, field='episode_rewards', title='rewards')
        self.histogram_all(runs, field='episode_lengths', title='lengths')
        self.describe_all(runs, field='episode_lengths', title='lengths')
        self.histogram_all(runs, field='episode_constraints', title='constraints', preprocess=lambda c: [sum(e) for e in c])
        self.describe_all(runs, field='episode_constraints', title='constraints', preprocess=lambda c: [sum(e) for e in c])
        plt.show()

    def compare(self, runs_directories_a, runs_directories_b):
        runs_a = {self.suffix(directory): MonitorV2.load_results(directory) for directory in runs_directories_a}
        runs_b = {self.suffix(directory): MonitorV2.load_results(directory) for directory in runs_directories_b}
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        self.plot_all(runs_a, field='episode_rewards', title='rewards', axes=ax1)
        self.plot_all(runs_b, field='episode_rewards', title='rewards', axes=ax2)
        plt.show()

    def histogram_all(self, runs, field, title, axes=None, preprocess=None):
        dirs = [directory for directory in runs.keys() if field in runs[directory]]
        data = [runs[directory][field] for directory in dirs]
        if preprocess:
            data = [preprocess(d) for d in data]
        data = [d[self.episodes_range[0]:self.episodes_range[1]] for d in data]
        axes = self.histogram(data, title=title, label=dirs, axes=axes)
        if axes:
            axes.legend()
            axes.grid()
        return axes

    @staticmethod
    def histogram(data, title, label, axes=None):
        if not axes:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.set_title('Histogram of {}'.format(title))
            axes.set_xlabel(title.capitalize())
            axes.set_ylabel('Frequency')
        weights = [np.ones(np.size(x))/np.size(x) for x in data]
        axes.hist(data, weights=weights, label=label, rwidth=1)
        return axes

    def plot_all(self, runs, field, title, axes=None):
        for directory, content in runs.items():
            axes = self.plot(content[field][self.episodes_range[0]:self.episodes_range[1]],
                             title=title, label=directory, axes=axes, averaged=False)
        if axes:
            axes.set_prop_cycle(None)
        for directory, content in runs.items():
            axes = self.plot(content[field][self.episodes_range[0]:self.episodes_range[1]],
                             title=title, label=directory, axes=axes, averaged=True)
        if axes:
            axes.legend()
            axes.grid()
        return axes

    def plot(self, data, title, label, axes=None, averaged=None):
        """
            Plot a data series
        :param data: a series
        :param title: fig title
        :param label: curve label
        :param axes:
        :param averaged:
        :return:
        """
        if not axes:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.set_title('History of {}'.format(title))
            axes.set_xlabel('Runs')
            axes.set_ylabel(title.capitalize())
        # Normal plot
        if averaged is None:
            axes.plot(np.arange(np.size(data)), data, label=label)
        # Averaged data plot
        elif averaged and np.size(data) > self.WINDOW:
            axes.plot(np.arange(np.size(data)), pd.Series(data).rolling(window=self.WINDOW).mean(), label=label)
        # Noisy data plot
        else:
            axes.plot(np.arange(np.size(data)), data, label=None, lw=3, alpha=.25)
        return axes

    def describe_all(self, runs, field, title, preprocess=None):
        print('---', title, '---')
        for directory, content in runs.items():
            if field not in content:
                continue
            data = content[field]
            if preprocess:
                data = preprocess(data)
            statistics = stats.describe(data[self.episodes_range[0]:self.episodes_range[1]])
            std = np.sqrt(statistics.variance) if not np.isnan(statistics.variance) else 0
            print(directory, '\t mean +/- std = {:.2f} +/- {:.2f} \t [min, max] = [{:.2f}, {:.2f}]'.format(
                statistics.mean, std, statistics.minmax[0], statistics.minmax[1]))
