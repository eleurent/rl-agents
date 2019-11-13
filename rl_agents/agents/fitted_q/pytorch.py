import logging
import pickle

from rl_agents.agents.common.optimizers import optimizer_factory
from rl_agents.agents.common.utils import get_memory, load_pytorch
from rl_agents.agents.deep_q_network.pytorch import DQNAgent
from rl_agents.agents.fitted_q.abstract import AbstractFTQAgent

logger = logging.getLogger(__name__)


class FTQAgent(AbstractFTQAgent, DQNAgent):
    def __init__(self, env, config=None):
        load_pytorch()
        super(FTQAgent, self).__init__(env, config)

    def initialize_model(self):
        self.value_net.reset()
        self.optimizer = optimizer_factory(self.config["optimizer"]["type"],
                                           self.value_net.parameters(),
                                           **self.config["optimizer"])

    def update_target_network(self):
        self.target_net.load_state_dict(self.value_net.state_dict())

    def save(self, filename):
        path = super().save(filename)
        samples_dataset_filename = filename.with_suffix(".data")
        with open(samples_dataset_filename, 'wb') as f:
            pickle.dump(self.memory.memory, f)
        logger.info("Saved a replay memory of length {}".format(len(self.memory)))
        return path

    def load(self, filename):
        path = super().load(filename)
        dataset_filename = filename.with_suffix(".data")
        with open(dataset_filename, 'rb') as f:
            self.memory.memory = pickle.load(f)
        logger.info("Loaded a replay memory of length {}".format(len(self.memory)))
        return path

    def log_memory(self, step):
        self.writer.add_scalar('agent/gpu_memory', sum(get_memory()), step)
