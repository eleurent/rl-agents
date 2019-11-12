"""
Usage:
  experiments evaluate <environment> <agent> (--train|--test) [options]
  experiments benchmark <benchmark> (--train|--test) [options]
  experiments -h | --help

Options:
  -h --help              Show this screen.
  --episodes <count>     Number of episodes [default: 5].
  --no-display           Disable environment, agent, and rewards rendering.
  --name-from-config     Name the output folder from the corresponding config files
  --processes <count>    Number of running processes [default: 4].
  --recover              Load model from the latest checkpoint.
  --recover-from <file>  Load model from a given checkpoint.
  --seed <str>           Seed the environments and agents.
  --train                Train the agent.
  --test                 Test the agent.
  --verbose              Set log level to debug instead of info.
  --repeat <times>       Repeat several times [default: 1].
"""
import datetime
import os
from pathlib import Path
import gym
import json
from docopt import docopt
from itertools import product
from multiprocessing.pool import Pool

from rl_agents.trainer import logger
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

BENCHMARK_FILE = 'benchmark_summary'
LOGGING_CONFIG = 'configs/logging.json'
VERBOSE_CONFIG = 'configs/verbose.json'


def main():
    opts = docopt(__doc__)
    if opts['evaluate']:
        for _ in range(int(opts['--repeat'])):
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
    logger.configure(LOGGING_CONFIG)
    if options['--verbose']:
        logger.configure(VERBOSE_CONFIG)
    env = load_environment(environment_config)
    agent = load_agent(agent_config, env)
    run_directory = None
    if options['--name-from-config']:
        run_directory = "{}_{}_{}".format(Path(agent_config).with_suffix('').name,
                                  datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                                  os.getpid())
    options['--seed'] = int(options['--seed']) if options['--seed'] is not None else None
    evaluation = Evaluation(env,
                            agent,
                            run_directory=run_directory,
                            num_episodes=int(options['--episodes']),
                            sim_seed=options['--seed'],
                            recover=options['--recover'] or options['--recover-from'],
                            display_env=not options['--no-display'],
                            display_agent=not options['--no-display'],
                            display_rewards=not options['--no-display'])
    if options['--train']:
        evaluation.train()
    elif options['--test']:
        evaluation.test()
    else:
        evaluation.close()
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
    generate_agent_configs(benchmark_config)
    experiments = product(benchmark_config['environments'], benchmark_config['agents'], [options])

    # Run evaluations
    with Pool(processes=int(options['--processes'])) as pool:
        results = pool.starmap(evaluate, experiments)

    # Clean temporary config files
    generate_agent_configs(benchmark_config, clean=True)

    # Write evaluations summary
    benchmark_filename = os.path.join(Evaluation.OUTPUT_FOLDER, '{}_{}.{}.json'.format(
        BENCHMARK_FILE, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid()))
    with open(benchmark_filename, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)
        gym.logger.info('Benchmark done. Summary written in: {}'.format(benchmark_filename))


def generate_agent_configs(benchmark_config, clean=False):
    """
        Generate several agent configurations from:
        - a "base_agent" configuration path field
        - a "key" field referring to a parameter that should vary
        - a "values" field listing the values of the parameter taken for each agent

        Created agent configurations will be stored in temporary file, that can be removed after use by setting the
        argument clean=True.
    :param benchmark_config: a benchmark configuration
    :param clean: should the temporary agent configurations files be removed
    :return the updated benchmark config
    """
    if "base_agent" in benchmark_config:
        with open(benchmark_config["base_agent"], 'r') as f:
            base_config = json.load(f)
            configs = [dict(base_config, **{benchmark_config["key"]: value})
                       for value in benchmark_config["values"]]
            paths = [Path(benchmark_config["base_agent"]).parent / "bench_{}={}.json".format(benchmark_config["key"], value)
                     for value in benchmark_config["values"]]
            if clean:
                [path.unlink() for path in paths]
            else:
                [json.dump(config, path.open('w')) for config, path in zip(configs, paths)]
            benchmark_config["agents"] = paths
    return benchmark_config


if __name__ == "__main__":
    main()
