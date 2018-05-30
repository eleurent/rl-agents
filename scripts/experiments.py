"""
Usage:
  experiments evaluate <environment_config> <agent_config> (--train|--test) [options]
  experiments -h | --help

Options:
  -h --help           Show this screen.
  --train             Train the agent, low monitoring.
  --test              Test the agent, high monitoring.
  --episodes <count>  Number of episodes [default: 5].
  --analyze           Automatically analyze the experiment results.
"""
import gym
import json
from docopt import docopt

from rl_agents.agents.common import agent_factory
from rl_agents.trainer.analyzer import RunAnalyzer
from rl_agents.trainer.simulation import Simulation


def main():
    opts = docopt(__doc__)
    if opts['evaluate']:
        evaluate(opts)


def evaluate(options):
    """
        Evaluate an agent interacting with an environment
    """
    gym.logger.set_level(gym.logger.INFO)
    env = load_environment(options)
    agent = load_agent(options, env)
    sim = Simulation(env, agent, num_episodes=int(options['--episodes']))
    if options['--train']:
        sim.train()
    elif options['--test']:
        sim.test()
    else:
        sim.close()
    if options['--analyze']:
        RunAnalyzer([sim.monitor.directory])


def load_environment(options):
    """
        Load an environment from its configuration file.
    """
    with open(options['<environment_config>']) as f:
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


def load_agent(options, env):
    """
        Load an agent from its configuration file, and environment.
    """
    # Load agent
    with open(options['<agent_config>']) as f:
        agent_config = json.loads(f.read())
    return agent_factory(env, agent_config)


if __name__ == "__main__":
    main()
