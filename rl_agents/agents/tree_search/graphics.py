import pygame
import matplotlib as mpl
import matplotlib.cm as cm


class MCTSGraphics(object):
    """
        Graphical visualization of the MCTSAgent tree.
    """
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)

    @classmethod
    def display(cls, agent, surface):
        """
            Display the whole tree of an MCTSAgent.

        :param agent: the MCTSAgent to be displayed
        :param surface: the pygame surface on which the agent is displayed
        """
        cell_size = (surface.get_width() // agent.mcts.config['max_depth'], surface.get_height())
        pygame.draw.rect(surface, cls.BLACK, (0, 0, surface.get_width(), surface.get_height()), 0)
        cls.display_node(agent.mcts.root, agent.env.action_space, surface, (0, 0), cell_size,
                         temperature=agent.mcts.config['temperature'], depth=0, selected=True)

        actions = agent.mcts.get_plan()
        font = pygame.font.Font(None, 13)
        text = font.render('-'.join(map(str, actions)), 1, (10, 10, 10), (255, 255, 255))
        surface.blit(text, (1, surface.get_height()-10))

    @classmethod
    def display_node(cls, node, action_space, surface, origin, size,
                     temperature=0,
                     depth=0,
                     selected=False,
                     display_exploration=False,
                     display_count=False,
                     display_text=True,
                     display_prior=True):
        """
            Display an MCTS node at a given position on a surface.

        :param node: the MCTS node to be displayed
        :param action_space: the environment action space
        :param surface: the pygame surface on which the node is displayed
        :param origin: the location of the node on the surface [px]
        :param size: the size of the node on the surface [px]
        :param temperature: the temperature used for exploration bonus visualization
        :param depth: the depth of the node in the tree
        :param selected: whether the node is within a selected branch of the tree
        :param display_exploration: display the exploration bonus
        :param display_count: display the visitation count of the node
        :param display_text: display a text showing the value and visitation count of the node
        :param display_prior: show the prior probability of each action
        """
        # Display node value
        cmap = cm.jet_r
        norm = mpl.colors.Normalize(vmin=-2, vmax=2)
        color = cmap(norm(node.value), bytes=True)
        pygame.draw.rect(surface, color, (origin[0], origin[1], size[0], size[1]), 0)

        # Add exploration bonus
        if display_exploration:
            norm = mpl.colors.Normalize(vmin=-2, vmax=2)
            color = cmap(norm(node.selection_strategy()), bytes=True)
            pygame.draw.polygon(surface, color, [(origin[0], origin[1] + size[1]),
                                                 (origin[0] + size[0], origin[1]),
                                                 (origin[0] + size[0], origin[1] + size[1])], 0)

        # Add node count
        if display_count and depth < 3:
            norm = mpl.colors.Normalize(vmin=0, vmax=100)
            color = cmap(norm(node.count), bytes=True)
            pygame.draw.rect(surface, color, (origin[0], origin[1], 5, 5), 0)

        # Add selection display
        if selected:
            pygame.draw.rect(surface, cls.RED, (origin[0], origin[1], size[0], size[1]), 1)

        if display_text and depth < 2:
            font = pygame.font.Font(None, 13)
            text = "{:.2f} / {:.2f} / {}".format(node.value, node.selection_strategy(temperature), node.count)
            if display_prior:
                text += " / {:.2f}".format(node.prior)
            text = font.render(text,
                               1, (10, 10, 10), (255, 255, 255))
            surface.blit(text, (origin[0]+1, origin[1]+1))

        # Recursively display children nodes
        best_action = node.select_action(temperature=0)
        for a in range(action_space.n):
            if a in node.children:
                action_selected = (selected and (a == best_action))
                cls.display_node(node.children[a], action_space, surface,
                                 (origin[0]+size[0], origin[1]+a*size[1]/action_space.n),
                                 (size[0], size[1]/action_space.n),
                                 depth=depth+1, temperature=temperature, selected=action_selected)


class RobustMCTSGraphics(object):
    @classmethod
    def display(cls, agent, surface):
        cell_size = (surface.get_width() // len(agent.agents), surface.get_height())
        pygame.draw.rect(surface, MCTSGraphics.BLACK, (0, 0, surface.get_width(), surface.get_height()), 0)
        for i, sub_agent in enumerate(agent.agents):
            sub_cell_size = (cell_size[0] // sub_agent.mcts.config["max_depth"], cell_size[1])
            MCTSGraphics.display_node(sub_agent.mcts.root, sub_agent.env.action_space, surface,
                                      (i*cell_size[0], 0), sub_cell_size,
                                      temperature=sub_agent.mcts.config['temperature'], depth=0, selected=True)