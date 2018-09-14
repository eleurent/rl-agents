import numpy as np


class Node(object):
    """
        A tree node
    """

    def __init__(self, parent, planner):
        """
            New node.

        :param parent: its parent node
        :param planner: the planner using the node
        """
        self.parent = parent
        self.planner = planner
        self.children = {}
        self.count = 0
        self.value = 0

    def get_value(self):
        return self.value

    def expand(self, branching_factor):
        for a in range(branching_factor):
            self.children[a] = type(self)(self, self.planner)

    @staticmethod
    def breadth_first_search(root, operator=None, condition=None):
        """
            Breadth-first search of all paths to nodes that meet a given condition

        :param root: starting node
        :param operator: will be applied to all traversed nodes
        :param condition: nodes meeting that condition will be returned
        :return: list of paths to nodes that met the condition
        """
        queue = [(root, [])]
        while queue:
            (node, path) = queue.pop(0)
            for next_key, next_node in node.children.items():
                if (condition is None) or condition(next_node):
                        returned = operator(next_node, path + [next_key]) if operator else (next_node, path + [next_key])
                        yield returned
                if (condition is None) or not condition(next_node):
                        queue.append((next_node, path + [next_key]))

    def is_leaf(self):
        return not self.children

    def path(self):
        node = self
        path = []
        while node.parent:
            for a in node.parent.children:
                if node.parent.children[a] == node:
                    path.append(a)
                    break
            node = node.parent
        return reversed(path)

    @staticmethod
    def all_argmax(x):
        m = np.amax(x)
        return np.nonzero(x == m)[0]

    def random_argmax(self, x):
        """
            Randomly tie-breaking arg max
        :param x: an array
        :return: a random index among the maximums
        """
        indices = Node.all_argmax(x)
        return self.planner.np_random.choice(indices)

    def __str__(self, level=0):
        return str(self.value)
        ret = "\t" * level + repr(self.value) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'
