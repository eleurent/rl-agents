from abc import ABC, abstractmethod
import numpy as np
from gym import spaces

from rl_agents.agents.common.abstract import AbstractStochasticAgent
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.memory import ReplayMemory, Transition


class AbstractDQNAgent(AbstractStochasticAgent, ABC):
    def __init__(self, env, config=None):
        super(AbstractDQNAgent, self).__init__(config)
        self.env = env
        assert isinstance(env.action_space, spaces.Discrete), "Only compatible with Discrete action spaces."
        self.config["model"]["in"] = int(np.prod(env.observation_space.shape))
        self.config["model"]["out"] = env.action_space.n
        self.memory = ReplayMemory(self.config)
        self.exploration_policy = exploration_factory(self.config["exploration"], self.env.action_space)
        self.training = True
        self.previous_state = None

    @classmethod
    def default_config(cls):
        return dict(model=dict(type="DuelingNetwork",
                               layers=[100, 100]),
                    optimizer=dict(type="ADAM",
                                   lr=5e-4,
                                   weight_decay=0,
                                   k=5),
                    loss_function="smooth_l1",
                    memory_capacity=50000,
                    batch_size=100,
                    gamma=0.99,
                    device="cuda:best",
                    exploration=dict(method="EpsilonGreedy"),
                    target_update=1)

    def record(self, state, action, reward, next_state, done, info):
        """
            Record a transition by performing a Deep Q-Network iteration

            - push the transition into memory
            - sample a minibatch
            - compute the bellman residual loss over the minibatch
            - perform one gradient descent step
            - slowly track the policy network with the target network
        :param state: a state
        :param action: an action
        :param reward: a reward
        :param next_state: a next state
        :param done: whether state is terminal
        """
        if not self.training:
            return
        self.memory.push(state, action, reward, next_state, done, info)
        batch = self.sample_minibatch()
        if batch:
            loss, _, _ = self.compute_bellman_residual(batch)
            self.step_optimizer(loss)
            self.update_target_network()

    def act(self, state):
        """
            Act according to the state-action value model and an exploration policy
        :param state: current state
        :return: an action
        """
        self.previous_state = state
        values = self.get_state_action_values(state)
        self.exploration_policy.update(values, step_time=True)
        return self.exploration_policy.sample()

    def sample_minibatch(self):
        if len(self.memory) < self.config["batch_size"]:
            return None
        transitions = self.memory.sample(self.config["batch_size"])
        return Transition(*zip(*transitions))

    def update_target_network(self):
        self.steps += 1
        if self.steps % self.config["target_update"] == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())

    @abstractmethod
    def compute_bellman_residual(self, batch, target_state_action_value=None):
        """
            Compute the Bellman Residual Loss over a batch
        :param batch: batch of transitions
        :param target_state_action_value: if provided, acts as a target (s,a)-value
                                          if not, it will be computed from batch and model (Double DQN target)
        :return: the loss over the batch, and the computed target
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch_state_values(self, states):
        """
        Get the state values of several states
        :param states: [s1; ...; sN] an array of states
        :return: values, actions:
                 - [V1; ...; VN] the array of the state values for each state
                 - [a1*; ...; aN*] the array of corresponding optimal action indexes for each state
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch_state_action_values(self, states):
        """
        Get the state-action values of several states
        :param states: [s1; ...; sN] an array of states
        :return: values:[[Q11, ..., Q1n]; ...] the array of all action values for each state
        """
        raise NotImplementedError

    def get_state_value(self, state):
        """
        :param state: s, an environment state
        :return: V, its state-value
        """
        values, actions = self.get_batch_state_values([state])
        return values[0], actions[0]

    def get_state_action_values(self, state):
        """
        :param state: s, an environment state
        :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions
        """
        return self.get_batch_state_action_values([state])[0]

    def step_optimizer(self, loss):
        raise NotImplementedError

    def seed(self, seed=None):
        return self.exploration_policy.seed(seed)

    def reset(self):
        pass

    def set_writer(self, writer):
        super().set_writer(writer)
        try:
            self.exploration_policy.set_writer(writer)
        except AttributeError:
            pass

    def action_distribution(self, state):
        self.previous_state = state
        values = self.get_state_action_values(state)
        self.exploration_policy.update(values, step_time=False)
        return self.exploration_policy.get_distribution()

    def set_time(self, time):
        self.exploration_policy.set_time(time)

    def eval(self):
        self.training = False
        self.config['exploration']['method'] = "Greedy"
        self.exploration_policy = exploration_factory(self.config["exploration"], self.env.action_space)
