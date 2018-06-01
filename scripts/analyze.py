"""
Usage:
  analyze run <run_folder>... [options]
  analyze benchmark <benchmark_file> [options]
  analyze -h | --help

Options:
  -h --help            Show this screen.
  --last <episodes>  Use only the N last episodes of the runs.
"""
import json
from docopt import docopt

from rl_agents.trainer.analyzer import RunAnalyzer


def main():
    opts = docopt(__doc__)
    if opts['--last']:
        episodes_range = [-int(opts['--last']), None]
    else:
        episodes_range = [None, None]
    if opts['run']:
        RunAnalyzer(opts['<run_folder>'], episodes_range=episodes_range)
    elif opts['benchmark']:
        with open(opts['<benchmark_file>'], 'r') as f:
            RunAnalyzer(json.loads(f.read()), episodes_range=episodes_range)


if __name__ == '__main__':
    main()


