import os

import numpy as np
from abc import ABC
import logging

from rl_agents.agents.common.memory import Transition
from rl_agents.agents.deep_q_network.abstract import AbstractDQNAgent

logger = logging.getLogger(__name__)


class AbstractFTQAgent(AbstractDQNAgent, ABC):
    def __init__(self, env, config=None):
        super(AbstractFTQAgent, self).__init__(env, config)
        self.batched = True
        self.iterations_time = 0
        self.regression_time = 0
        self.batch_time = 0

    @classmethod
    def default_config(cls):
        cfg = super(AbstractFTQAgent, cls).default_config()
        cfg.update({"value_iteration_epochs": "from-gamma",
                    "regression_epochs": 50,
                    "processes": os.cpu_count(),
                    "constraint_penalty": 0})
        return cfg

    def record(self, state, action, reward, next_state, done, info):
        """
            Record a transition by performing a Fitted-Q iteration

            - push the transition into memory

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

    def update(self):
        """
            Updates the value model.
                - perform N value iteration steps Qk -> Qk+1, ie:
                - compute the Bellman residual loss over the batch
                - Minimize it through M gradient descent steps
        """
        batch = self.sample_minibatch()
        batch = self._add_constraint_penalty(batch)
        self.batch_time += 1
        if self.writer:
            self.writer.add_scalar('agent/batch_size', len(batch.state), self.batch_time)
        # Optimize model on batch
        value_iteration_epochs = self.config["value_iteration_epochs"] or int(3 / (1 - self.config["gamma"]))
        self.initialize_model()
        for epoch in range(value_iteration_epochs):
            self.update_target_network()
            delta, target, batch = self.compute_bellman_residual(batch)
            self.initialize_model()
            logger.debug("Bellman residual at iteration {} is {}".format(epoch, delta))
            if self.writer:
                self.writer.add_scalar('agent/bellman_residual', delta, self.iterations_time)
                self.log_memory(self.iterations_time)
                self.iterations_time += 1

            for step in range(self.config["regression_epochs"]):
                batch = self.sample_minibatch()

                loss, _, _ = self.compute_bellman_residual(batch)
                self.step_optimizer(loss)
                if self.writer:
                    if self.regression_time % 10 == 0:
                        self.writer.add_scalar('agent/regression_loss', loss, self.regression_time)
                    self.regression_time += 1

    def sample_minibatch(self):
        """
            Sample a batch of transitions from memory.
        :return: a batch of the whole memory
        """
        transitions = self.memory.sample(64)
        # transitions = self.memory.sample(len(self.memory))
        return Transition(*zip(*transitions))

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

    def reset(self):
        super().reset()
        self.iterations_time = 0
        self.regression_time = 0
        self.batch_time = 0

    def initialize_model(self):
        raise NotImplementedError

    def log_memory(self, log_memory):
        raise NotImplementedError

