import copy
import importlib
import json
import logging
import gym

logger = logging.getLogger(__name__)


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
    # Load agent from file
    if not isinstance(agent_path, dict):
        with open(agent_path) as f:
            agent_config = json.loads(f.read())
    else:
        agent_config = agent_path

    return agent_factory(env, agent_config)


def load_environment(env_config):
    """
        Load an environment from a configuration file.

    :param env_config: the configuration, or path to the environment configuration file
    :return: the environment
    """
    # Load the environment config from file
    if not isinstance(env_config, dict):
        with open(env_config) as f:
            env_config = json.loads(f.read())

    # Make the environment
    if env_config.get("import_module", None):
        __import__(env_config["import_module"])
    try:
        env = gym.make(env_config['id'])
        # Save env module in order to be able to import it again
        env.import_module = env_config.get("import_module", None)
    except KeyError:
        raise ValueError("The gym register id of the environment must be provided")
    except gym.error.UnregisteredEnv:
        # The environment is unregistered.
        print("import_module", env_config["import_module"])
        raise gym.error.UnregisteredEnv('Environment {} not registered. The environment module should be specified by '
                                        'the "import_module" key of the environment configuration'.format(
                                            env_config['id']))

    # Configure the environment, if supported
    try:
        env.unwrapped.configure(env_config)
        # Reset the environment to ensure configuration is applied
        env.reset()
    except AttributeError as e:
        logger.info("This environment does not support configuration. {}".format(e))
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
            try:
                preprocessor = getattr(env.unwrapped, preprocessor_config["method"])
                if "args" in preprocessor_config:
                    env = preprocessor(preprocessor_config["args"])
                else:
                    env = preprocessor()
            except AttributeError:
                logger.warn("The environment does not have a {} method".format(preprocessor_config["method"]))
        else:
            logger.error("The method is not specified in ", preprocessor)
    return env


def safe_deepcopy_env(obj):
    """
        Perform a deep copy of an environment but without copying its viewer.
    """
    cls = obj.__class__
    result = cls.__new__(cls)
    memo = {id(obj): result}
    for k, v in obj.__dict__.items():
        if k not in ['viewer', 'automatic_rendering_callback', 'grid_render']:
            if isinstance(v, gym.Env):
                setattr(result, k, safe_deepcopy_env(v))
            else:
                setattr(result, k, copy.deepcopy(v, memo=memo))
        else:
            setattr(result, k, None)
    return result
