import numpy as np
from abc import ABC
from gym import logger

from rl_agents.agents.common.memory import Transition
from rl_agents.agents.deep_q_network.abstract import AbstractDQNAgent


class AbstractFTQAgent(AbstractDQNAgent, ABC):
    def __init__(self, env, config=None):
        super(AbstractFTQAgent, self).__init__(env, config)

    @classmethod
    def default_config(cls):
        cfg = super(AbstractFTQAgent, cls).default_config()
        cfg.update({"value_iteration_epochs": "from-gamma",
                    "regression_epochs": 50,
                    "constraint_penalty": 0})
        return cfg

    def record(self, state, action, reward, next_state, done, info):
        """
            Record a transition by performing a Fitted-Q iteration

            - push the transition into memory
            - when enough experience is acquired, sample a batch
            - perform N value iteration steps Qk -> Qk+1, ie:
                - compute the Bellman residual loss over the batch
                - Minimize it through M gradient descent steps
        :param state: a state
        :param action: an action
        :param reward: a reward
        :param next_state: a next state
        :param done: whether state is terminal
        :param info: information about the environment
        """
        if not self.training:
            return
        # Store transition to memory
        self.memory.push(state, action, reward, next_state, done, info)
        batch = self.sample_minibatch()
        if not batch:
            return
        batch = self._add_constraint_penalty(batch)
        # Optimize model on batch
        value_iteration_epochs = self.config["value_iteration_epochs"] or int(3 / (1 - self.config["gamma"]))
        self.initialize_model()
        for epoch in range(value_iteration_epochs):
            self.update_target_network()
            delta, target = self.compute_bellman_residual(batch)
            self.initialize_model()
            logger.info("Bellman residual at iteration {} on batch {} is {}".format(epoch, len(batch.reward), delta))
            for _ in range(self.config["regression_epochs"]):
                loss, _ = self.compute_bellman_residual(batch, target)
                self.step_optimizer(loss)

    def sample_minibatch(self):
        """
            Sample a batch of transitions from memory.
            This only happens
                - when the memory is full
                - at some intermediate memory lengths
            Otherwise, the returned batch is empty
        :return: a batch of the whole memory
        """
        if self.memory.is_full():
            logger.info("Memory is full, switching to evaluation mode.")
            self.eval()
            transitions = self.memory.sample(len(self.memory))
            return Transition(*zip(*transitions))
        elif len(self.memory) % self.config["batch_size"] == 0:
            transitions = self.memory.sample(len(self.memory))
            return Transition(*zip(*transitions))
        else:
            return None

    def _add_constraint_penalty(self, batch):
        """
            If a constraint penalty is specified, modify the batch rewards to include this penalty
        :param batch: a batch of transitions
        :return: the modified batch
        """
        if self.config["constraint_penalty"] and "constraint" in batch.info[0]:
            batch = batch._replace(reward=tuple(np.array(batch.reward) + self.config["constraint_penalty"] *
                                                np.array([info["constraint"] for info in batch.info])))
        return batch

