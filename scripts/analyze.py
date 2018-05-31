"""
Usage:
  analyze run <run_folder>...
  analyze benchmark <benchmark_file>
  analyze -h | --help

Options:
  -h --help           Show this screen.
"""
from docopt import docopt
from rl_agents.trainer.analyzer import RunAnalyzer
from rl_agents.trainer.benchmark import Benchmark


def main():
    opts = docopt(__doc__)
    if opts['run']:
        RunAnalyzer(opts['<run_folder>'])
    elif opts['benchmark']:
        Benchmark.open_runs_summary(opts['<benchmark_file>'])


if __name__ == '__main__':
    main()


