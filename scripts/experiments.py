"""
Usage:
  experiments evaluate <environment> <agent> (--train|--test)
                                             [--episodes <count>]
                                             [--seed <str>]
                                             [--analyze]
  experiments benchmark <benchmark> (--train|--test)
                                    [--processes <count>]
                                    [--episodes <count>]
                                    [--seed <str>]
  experiments -h | --help

Options:
  -h --help            Show this screen.
  --analyze            Automatically analyze the experiment results.
  --episodes <count>   Number of episodes [default: 5].
  --processes <count>  Number of running processes [default: 4].
  --seed <str>         Seed the environments and agents.
  --train              Train the agent.
  --test               Test the agent.
"""

import gym
import json
from docopt import docopt
from itertools import product
from multiprocessing.pool import Pool

from rl_agents.agents.common import agent_factory
from rl_agents.trainer.analyzer import RunAnalyzer
from rl_agents.trainer.evaluation import Evaluation


def main():
    opts = docopt(__doc__)
    if opts['evaluate']:
        evaluate(opts['<environment>'], opts['<agent>'], opts)
    elif opts['benchmark']:
        benchmark(opts)


def evaluate(environment_config, agent_config, options):
    """
        Evaluate an agent interacting with an environment.

    :param environment_config: the path of the environment configuration file
    :param agent_config: the path of the agent configuration file
    :param options: the evaluation options
    """
    gym.logger.set_level(gym.logger.INFO)
    env = Evaluation.load_environment(environment_config)
    agent = Evaluation.load_agent(agent_config, env)
    evaluation = Evaluation(env, agent, num_episodes=int(options['--episodes']), sim_seed=options['--seed'])
    if options['--train']:
        evaluation.train()
    elif options['--test']:
        evaluation.test()
    else:
        evaluation.close()
    if options['--analyze']:
        RunAnalyzer([evaluation.monitor.directory])
    return evaluation.monitor.directory


def benchmark(options):
    """
        Run the evaluations of several agents interacting in several environments.

    The evaluations are dispatched over several processes.
    The benchmark configuration file should look like this:
    {
        "environments": ["path/to/env1.json", ...],
        "agents: ["path/to/agent1.json", ...]
    }

    :param options: the evaluation options, containing the path to the benchmark configuration file.
    """
    with open(options['<benchmark>']) as f:
        benchmark_config = json.loads(f.read())
    experiments = product(benchmark_config['environments'], benchmark_config['agents'], [options])
    with Pool(processes=int(options['--processes'])) as pool:
        results = pool.starmap(evaluate, experiments)
    gym.logger.info('Generated runs: {}'.format(results))


if __name__ == "__main__":
    main()
