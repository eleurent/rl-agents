from __future__ import division, print_function

from rl_agents.agents.dqn.abstract import AbstractDQNAgent
from rl_agents.agents.dqn.graphics import DQNGraphics
from rl_agents.agents.dynamic_programming.graphics import ValueIterationGraphics
from rl_agents.agents.dynamic_programming.ttc_vi import TTCVIAgent
from rl_agents.agents.dynamic_programming.value_iteration import ValueIterationAgent
from rl_agents.agents.tree_search.graphics import MCTSGraphics, OneStepRobustMCTSGraphics, DiscreteRobustMCTSGraphics, \
    IntervalRobustMCTSGraphics
from rl_agents.agents.tree_search.mcts import MCTSAgent
from rl_agents.agents.tree_search.robust_mcts import DiscreteRobustMCTSAgent, IntervalRobustMCTS, OneStepRobustMCTS


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

        if isinstance(agent, DiscreteRobustMCTSAgent):
            DiscreteRobustMCTSGraphics.display(agent, agent_surface)
        elif isinstance(agent, MCTSAgent):
            MCTSGraphics.display(agent, agent_surface)
        elif isinstance(agent, AbstractDQNAgent):
            DQNGraphics.display(agent, agent_surface)
        elif isinstance(agent, ValueIterationAgent):
            ValueIterationGraphics.display(agent, agent_surface)
        elif isinstance(agent, OneStepRobustMCTS):
            OneStepRobustMCTSGraphics.display(agent, agent_surface)
        elif isinstance(agent, IntervalRobustMCTS):
            IntervalRobustMCTSGraphics.display(agent, agent_surface, sim_surface)
