from __future__ import division, print_function

from rl_agents.agents.dqn.abstract import AbstractDQNAgent
from rl_agents.agents.dqn.graphics import DQNGraphics
from rl_agents.agents.dynamic_programming.graphics import TTCVIGraphics
from rl_agents.agents.dynamic_programming.ttc_vi import TTCVIAgent
from rl_agents.agents.linear.graphics import LinearModelGraphics
from rl_agents.agents.linear.linear_model import LinearModelAgent
from rl_agents.agents.tree_search.graphics import MCTSGraphics, RobustMCTSGraphics
from rl_agents.agents.tree_search.mcts import MCTSAgent
from rl_agents.agents.tree_search.robust_mcts import RobustMCTSAgent


class AgentGraphics(object):
    """
        Graphical visualization of any Agent implementing AbstractAgent.
    """
    @classmethod
    def display(cls, agent, agent_surface, sim_surface=None):
        """
            Display an agent visualization on a pygame surface.

        :param agent: the agent to be displayed
        :param agent_surface: the pygame surface on which the agent is displayed
        :param sim_surface: the pygame surface on which the environment is displayed
        """

        if isinstance(agent, MCTSAgent):
            MCTSGraphics.display(agent, agent_surface)
        elif isinstance(agent, AbstractDQNAgent):
            DQNGraphics.display(agent, agent_surface)
        elif isinstance(agent, TTCVIAgent):
            TTCVIGraphics.display(agent, agent_surface)
        elif isinstance(agent, RobustMCTSAgent):
            RobustMCTSGraphics.display(agent, agent_surface)
        elif isinstance(agent, LinearModelAgent):
            LinearModelGraphics.display(agent, sim_surface)
