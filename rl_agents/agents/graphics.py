from __future__ import division, print_function

from rl_agents.agents.dqn.abstract import AbstractDQNAgent
from rl_agents.agents.dqn.graphics import DQNGraphics
from rl_agents.agents.dynamic_programming.graphics import TTCVIGraphics
from rl_agents.agents.dynamic_programming.ttc_vi import TTCVIAgent
from rl_agents.agents.tree_search.graphics import MCTSGraphics
from rl_agents.agents.tree_search.mcts import MCTSAgent


class AgentGraphics(object):
    """
        Graphical visualization of any Agent implementing AbstractAgent.
    """
    @classmethod
    def display(cls, agent, surface):
        """
            Display an agent visualization on a pygame surface.

        :param agent: the agent to be displayed
        :param surface: the pygame surface on which the agent is displayed
        :return:
        """

        if isinstance(agent, MCTSAgent):
            MCTSGraphics.display(agent, surface)
        elif isinstance(agent, AbstractDQNAgent):
            DQNGraphics.display(agent, surface)
        elif isinstance(agent, TTCVIAgent):
            TTCVIGraphics.display(agent, surface)


