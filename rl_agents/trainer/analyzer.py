import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from rl_agents.trainer.monitor import MonitorV2
from matplotlib.patches import Ellipse

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
        runs = self.preprocess(runs)
        self.plot_all(runs, field='episode_rewards', title='rewards')
        self.histogram_all(runs, field='discounted_rewards', title='rewards')
        self.describe_all(runs, field='discounted_rewards', title='rewards')
        self.histogram_all(runs, field='episode_lengths', title='lengths')
        self.describe_all(runs, field='episode_lengths', title='lengths')
        self.histogram_all(runs, field='discounted_costs', title='costs')
        self.describe_all(runs, field='discounted_costs', title='costs')
        self.compare(runs)
        plt.show()

    def preprocess(self, runs, gamma=0.9):
        for dir, data in runs.items():
            data["discounted_rewards"] = [np.sum([episode[t]*gamma**t for t in range(len(episode))])
                                          for episode in data["episode_rewards_"]]
            data["discounted_costs"] = [np.sum([episode[t]*gamma**t for t in range(len(episode))])
                                          for episode in data["episode_costs"]]
        return runs

    def compare(self, runs):
        if len(runs) < 2:
            return
        self.plot_all_confidence_ellipse(runs,
                                         "discounted_costs",
                                         "discounted_rewards",
                                         ["costs", "rewards"])

    def histogram_all(self, runs, field, title, axes=None):
        dirs = [directory for directory in runs.keys() if field in runs[directory]]
        data = [runs[directory][field] for directory in dirs]
        data = [d[self.episodes_range[0]:self.episodes_range[1]] for d in data]
        axes = self.histogram(data, title=title, labels=dirs, axes=axes)
        if axes:
            axes.legend()
            axes.grid()
        return axes

    @staticmethod
    def histogram(data, title, labels, axes=None):
        if not axes:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.set_title('Histogram of {}'.format(title))
            axes.set_xlabel(title.capitalize())
            axes.set_ylabel('Frequency')
        for x, label in zip(data, labels):
            sns.distplot(x, label=label, ax=axes)
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

    def plot_all_confidence_ellipse(self, runs, field_x, field_y, labels, axes=None):
        x, y = field_x, field_y
        if isinstance(field_x, str):
            x = lambda d: d[field_x]
        if isinstance(field_y, str):
            y = lambda d: d[field_y]
        for directory, content in list(runs.items()):
            axes = self.plot_confidence_ellipse(x=x(content)[self.episodes_range[0]:self.episodes_range[1]],
                                                y=y(content)[self.episodes_range[0]:self.episodes_range[1]],
                                                axes=axes,
                                                labels=labels,
                                                title=directory)
        # plt.legend(list(runs.keys()))
        axes.autoscale(True)
        return axes

    def plot_confidence_ellipse(self, x, y, axes, labels, title, sigma=2):
        if not axes:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.set_title(r'Confidence ellipses at ${}\sigma$'.format(sigma))
            axes.set_xlabel(labels[0])
            axes.set_ylabel(labels[1])
        lambda_, v = np.linalg.eig(np.cov(x, y) / len(x))
        lambda_ = np.sqrt(lambda_)
        ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                          width=lambda_[0] * sigma, height=lambda_[1] * sigma,
                          angle=np.rad2deg(np.arccos(v[0, 0])), alpha=0.4)
        axes.add_patch(ellipse)
        plt.scatter(np.mean(x), np.mean(y))
        axes.annotate(title, (np.mean(x), np.mean(y)))
        return axes
