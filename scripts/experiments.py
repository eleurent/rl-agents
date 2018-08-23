"""
Usage:
  experiments evaluate <environment> <agent> (--train|--test)
                                             [--episodes <count>]
                                             [--name-from-config]
                                             [--no-display]
                                             [--seed <str>]
                                             [--analyze]
  experiments benchmark <benchmark> (--train|--test)
                                    [--episodes <count>]
                                    [--name-from-config]
                                    [--no-display]
                                    [--seed <str>]
                                    [--analyze]
                                    [--processes <count>]
  experiments -h | --help

Options:
  -h --help            Show this screen.
  --analyze            Automatically analyze the experiment results.
  --episodes <count>   Number of episodes [default: 5].
  --no-display         Disable environment, agent, and rewards rendering.
  --name-from-config   Name the output folder from the corresponding config files
  --processes <count>  Number of running processes [default: 4].
  --seed <str>         Seed the environments and agents.
  --train              Train the agent.
  --test               Test the agent.
"""
import datetime
import os

import gym
import json
from docopt import docopt
from itertools import product
from multiprocessing.pool import Pool

from rl_agents.trainer.analyzer import RunAnalyzer
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common import load_agent, load_environment

BENCHMARK_FILE = 'benchmark_summary'


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
    env = load_environment(environment_config)
    agent = load_agent(agent_config, env)
    if options['--name-from-config']:
        directory = os.path.join(Evaluation.OUTPUT_FOLDER,
                                 os.path.basename(environment_config).split('.')[0],
                                 os.path.basename(agent_config).split('.')[0])
    else:
        directory = None
    options['--seed'] = int(options['--seed']) if options['--seed'] is not None else None
    evaluation = Evaluation(env,
                            agent,
                            directory=directory,
                            num_episodes=int(options['--episodes']),
                            sim_seed=options['--seed'],
                            display_env=not options['--no-display'],
                            display_agent=not options['--no-display'],
                            display_rewards=not options['--no-display'])
    if options['--train']:
        evaluation.train()
    elif options['--test']:
        evaluation.test()
    else:
        evaluation.close()
    if options['--analyze'] and not options['<benchmark>']:
        RunAnalyzer([evaluation.monitor.directory])
    return os.path.relpath(evaluation.monitor.directory)


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
    # Prepare experiments
    with open(options['<benchmark>']) as f:
        benchmark_config = json.loads(f.read())
    experiments = product(benchmark_config['environments'], benchmark_config['agents'], [options])

    # Run evaluations
    with Pool(processes=int(options['--processes'])) as pool:
        results = pool.starmap(evaluate, experiments)

    # Write evaluations summary
    benchmark_filename = os.path.join(Evaluation.OUTPUT_FOLDER, '{}_{}.{}.json'.format(
        BENCHMARK_FILE, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid()))
    with open(benchmark_filename, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)
        gym.logger.info('Benchmark done. Summary written in: {}'.format(benchmark_filename))

    if options['--analyze']:
        RunAnalyzer(results)


if __name__ == "__main__":
    main()
