import importlib
import json
import gym
from gym import logger


def agent_factory(environment, config):
    """
        Handles creation of agents.

    :param environment: the environment
    :param config: configuration of the agent, must contain a '__class__' key
    :return: a new agent
    """
    if "__class__" in config:
        path = config['__class__'].split("'")[1]
        module_name, class_name = path.rsplit(".", 1)
        agent_class = getattr(importlib.import_module(module_name), class_name)
        agent = agent_class(environment, config)
        return agent
    else:
        raise ValueError("The configuration should specify the agent __class__")


def load_agent(agent_path, env):
    """
        Load an agent from a configuration file.

    :param agent_path: the path to the agent configuration file
    :param env: the environment with which the agent interacts
    :return: the agent
    """
    # Load agent
    with open(agent_path) as f:
        agent_config = json.loads(f.read())
    return agent_factory(env, agent_config)


def load_environment(env_path):
    """
        Load an environment from a configuration file.

    :param env_path: the path to the environment configuration file
    :return: the environment
    """
    # Load the environment config from file
    with open(env_path) as f:
        env_config = json.loads(f.read())

    # Make the environment
    try:
        env = gym.make(env_config['id'])
    except KeyError:
        raise ValueError("The gym register id of the environment must be provided")
    except gym.error.UnregisteredEnv:
        # The environment is unregistered.
        gym.logger.warn("Environment {} not found".format(env_config['id']))
        # Automatic import of some modules to register them.
        if env_config['id'].startswith('obstacle'):
            gym.logger.info("Importing obstacle_env module")
            import obstacle_env
            env = gym.make(env_config['id'])
        elif env_config['id'].startswith('highway'):
            gym.logger.info("Importing highway_env module")
            import highway_env
            env = gym.make(env_config['id'])
        elif env_config['id'].startswith('finite-mdp'):
            gym.logger.info("Importing finite_mdp module")
            import finite_mdp
            env = gym.make(env_config['id'])
        else:
            raise gym.error.UnregisteredEnv("Unregistered environment, and neither highway_env, obstacle_env nor "
                                            "finite-mdp")

    # Configure the environment, if supported
    try:
        env.configure(env_config)
    except AttributeError:
        pass
    return env


def preprocess_env(env, preprocessor_configs):
    """
        Apply a series of pre-processes to an environment, before it is used by an agent.
    :param env: an environment
    :param preprocessor_configs: a list of preprocessor configs
    :return: a preprocessed copy of the environment
    """
    for preprocessor_config in preprocessor_configs:
        if "method" in preprocessor_config:
            preprocessor = getattr(env, preprocessor_config["method"])
            if "args" in preprocessor_config:
                env = preprocessor(preprocessor_config["args"])
            else:
                env = preprocessor()
        else:
            logger.error("Unknown environment preprocessor", preprocessor)
    return env
