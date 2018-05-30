from rl_agents.agents.exploration.boltzmann import Boltzmann
from rl_agents.agents.exploration.epsilon_greedy import EpsilonGreedy
from rl_agents.agents.exploration.greedy import Greedy


def exploration_factory(exploration_config, action_space):
    """
        Handles creation of exploration policies
    :param exploration_config: configuration dictionary of the policy, must contain a "method" key
    :param action_space: the environment action space
    :return: a new exploration policy
    """
    if exploration_config['method'] == 'Greedy':
        return Greedy(action_space, exploration_config)
    elif exploration_config['method'] == 'EpsilonGreedy':
        return EpsilonGreedy(action_space, exploration_config)
    elif exploration_config['method'] == 'Boltzmann':
        return Boltzmann(action_space, exploration_config)
    else:
        raise ValueError("Unknown exploration method")
