from __future__ import division, print_function

from rl_agents.agents.budgeted_ftq.agent import BFTQAgent
from rl_agents.agents.budgeted_ftq.graphics import BFTQGraphics
from rl_agents.agents.deep_q_network.abstract import AbstractDQNAgent
from rl_agents.agents.deep_q_network.graphics import DQNGraphics
from rl_agents.agents.dynamic_programming.graphics import ValueIterationGraphics
from rl_agents.agents.dynamic_programming.value_iteration import ValueIterationAgent
from rl_agents.agents.tree_search.abstract import AbstractTreeSearchAgent
from rl_agents.agents.tree_search.mdp_gape import MDPGapEAgent
from rl_agents.agents.tree_search.graphics import TreeGraphics, MCTSGraphics, DiscreteRobustPlannerGraphics, \
    IntervalRobustPlannerGraphics
from rl_agents.agents.tree_search.mcts import MCTSAgent
from rl_agents.agents.tree_search.robust import DiscreteRobustPlannerAgent, IntervalRobustPlannerAgent


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

        if isinstance(agent, AbstractDQNAgent):
            DQNGraphics.display(agent, agent_surface, sim_surface)
        elif isinstance(agent, BFTQAgent):
            BFTQGraphics.display(agent, agent_surface)
        elif isinstance(agent, ValueIterationAgent):
            ValueIterationGraphics.display(agent, agent_surface)
        elif isinstance(agent, MCTSAgent):
            MCTSGraphics.display(agent, agent_surface)
        elif isinstance(agent, IntervalRobustPlannerAgent):
            IntervalRobustPlannerGraphics.display(agent, agent_surface, sim_surface)
        elif isinstance(agent, DiscreteRobustPlannerAgent):
            DiscreteRobustPlannerGraphics.display(agent, agent_surface, sim_surface)
        elif isinstance(agent, MDPGapEAgent):
            pass
        elif isinstance(agent, AbstractTreeSearchAgent):
            TreeGraphics.display(agent, agent_surface)
