import importlib


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
