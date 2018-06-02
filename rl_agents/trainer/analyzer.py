import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from rl_agents.trainer.monitor import MonitorV2


class RunAnalyzer(object):
    WINDOW = 50

    def __init__(self, run_directories, episodes_range=[None, None]):
        self.base = os.path.commonprefix(run_directories) if len(run_directories) > 1 else ''
        self.episodes_range = episodes_range
        self.analyze(run_directories)

    def suffix(self, directory):
        return directory[len(self.base):]

    def analyze(self, run_directories):
        runs = {self.suffix(directory): MonitorV2.load_results(directory) for directory in run_directories}
        self.plot_all(runs, field='episode_rewards', title='rewards')
        self.describe_all(runs, field='episode_rewards', title='rewards')
        self.histogram_all(runs, field='episode_rewards', title='rewards')
        # self.histogram_all(runs, field='episode_avg_rewards', title='average rewards')
        self.histogram_all(runs, field='episode_lengths', title='lengths')
        plt.show()

    def compare(self, runs_directories_a, runs_directories_b):
        runs_a = {self.suffix(directory): MonitorV2.load_results(directory) for directory in runs_directories_a}
        runs_b = {self.suffix(directory): MonitorV2.load_results(directory) for directory in runs_directories_b}
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        self.plot_all(runs_a, field='episode_rewards', title='rewards', axes=ax1)
        self.plot_all(runs_b, field='episode_rewards', title='rewards', axes=ax2)
        plt.show()

    def histogram_all(self, runs, field, title, axes=None):
        dirs = list(runs.keys())
        data = [runs[directory][field][self.episodes_range[0]:self.episodes_range[1]] for directory in dirs]
        axes = self.histogram(data, title=title, label=dirs, axes=axes)
        axes.legend()
        axes.grid()
        return axes

    def histogram(self, data, title, label, axes=None):
        if not axes:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.set_title('Histogram of {}'.format(title))
            axes.set_xlabel(title.capitalize())
            axes.set_ylabel('Frequency')
        axes.hist(data, density=True, label=label)
        return axes

    def plot_all(self, runs, field, title, axes=None):
        for directory, manifest in runs.items():
            axes = self.plot(manifest[field], title=title, label=directory, axes=axes, averaged=False)
        axes.set_prop_cycle(None)
        for directory, manifest in runs.items():
            axes = self.plot(manifest[field], title=title, label=directory, axes=axes, averaged=True)
        axes.legend()
        axes.grid()
        return axes

    def plot(self, data, title, label, axes=None, averaged=None):
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
            means = np.hstack((np.nan*np.ones((self.WINDOW,)),
                               np.convolve(data, np.ones((self.WINDOW,)) / self.WINDOW, mode='valid')))
            axes.plot(np.arange(np.size(means)), means, label=label)
        # Noisy data plot
        else:
            axes.plot(np.arange(np.size(data)), data, label=None, lw=3, alpha=.25)
        return axes

    def describe_all(self, runs, field, title):
        print('---', title, '---')
        for directory, manifest in runs.items():
            statistics = stats.describe(manifest[field][self.episodes_range[0]:self.episodes_range[1]])
            print(directory, '{:.2f} +/- {:.2f}'.format(statistics.mean, np.sqrt(statistics.variance)))

    def scatter(self, xx, yy, title_x, title_y, label, figure=None):
        if not figure:
            figure = plt.figure()
            plt.grid(True)
        plt.scatter(xx, yy, label=label)
        plt.title('{} with respect to {}'.format(title_x, title_y))
        plt.xlabel(title_x.capitalize())
        plt.ylabel(title_y.capitalize())
        plt.show()
        return figure
