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
    env = load_environment(environment_config)
    agent = load_agent(agent_config, env)
    sim = Evaluation(env, agent,
                     num_episodes=int(options['--episodes']),
                     sim_seed=options['--seed'])
    if options['--train']:
        sim.train()
    elif options['--test']:
        sim.test()
    else:
        sim.close()
    if options['--analyze']:
        RunAnalyzer([sim.monitor.directory])
    return sim.monitor.directory


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


def load_environment(env_config):
    """
        Load an environment from a configuration file.

    :param env_config: the path to the environment configuration file
    :return: the environment
    """
    with open(env_config) as f:
        env_config = json.loads(f.read())
    try:
        env = gym.make(env_config['id'])
    except KeyError:
        raise ValueError("The gym register id of the environment must be provided")
    except gym.error.UnregisteredEnv:
        gym.logger.warn("Environment {} not found".format(env_config['id']))
        if env_config['id'].startswith('obstacle'):
            gym.logger.info("Importing obstacle_env module")
            import obstacle_env
            env = gym.make(env_config['id'])
        elif env_config['id'].startswith('highway'):
            gym.logger.info("Importing highway_env module")
            import highway_env
            env = gym.make(env_config['id'])
    return env


def load_agent(agent_config, env):
    """
        Load an agent from a configuration file.

    :param agent_config: the path to the agent configuration file
    :param env: the environment with which the agent interacts
    :return: the agent
    """
    # Load agent
    with open(agent_config) as f:
        agent_config = json.loads(f.read())
    return agent_factory(env, agent_config)


if __name__ == "__main__":
    main()
