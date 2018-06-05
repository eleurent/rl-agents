"""
Usage:
  analyze run <run_folder>... [options]
  analyze benchmark <benchmark_file> [options]
  analyze -h | --help

Options:
  -h --help           Show this screen.
  --first <episodes>  Use only the N first episodes of the runs.
  --last <episodes>   Use only the N last episodes of the runs.
"""
import json
from docopt import docopt
import gym

from rl_agents.trainer.analyzer import RunAnalyzer


def main():
    gym.logger.set_level(gym.logger.INFO)
    opts = docopt(__doc__)
    episodes_range = [None, None]
    if opts['--first']:
        episodes_range[1] = int(opts['--first'])
    if opts['--last']:
        episodes_range[0] = -int(opts['--last'])
    if opts['run']:
        RunAnalyzer(opts['<run_folder>'], episodes_range=episodes_range)
    elif opts['benchmark']:
        with open(opts['<benchmark_file>'], 'r') as f:
            RunAnalyzer(json.loads(f.read()), episodes_range=episodes_range)


if __name__ == '__main__':
    main()


