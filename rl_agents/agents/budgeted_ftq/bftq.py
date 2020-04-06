"""
    Adapted from the original implementation by Nicolas Carrara <https://github.com/ncarrara>.
"""
from rl_agents.agents.budgeted_ftq.graphics import plot_values_histograms, plot_frontier
from rl_agents.agents.common.utils import choose_device

__author__ = "Edouard Leurent"
__credits__ = ["Nicolas Carrara"]

from multiprocessing.pool import Pool
from pathlib import Path
import numpy as np
import torch
import logging

from rl_agents.agents.budgeted_ftq.greedy_policy import TransitionBFTQ, pareto_frontier, \
    optimal_mixture
from rl_agents.agents.common.optimizers import loss_function_factory, optimizer_factory
from rl_agents.utils import near_split
from rl_agents.agents.common.memory import ReplayMemory

logger = logging.getLogger(__name__)


class BudgetedFittedQ(object):
    def __init__(self, value_network, config, writer=None):
        self.config = config

        # Load configs
        self.betas_for_duplication = parse(self.config["betas_for_duplication"])
        self.betas_for_discretisation = parse(self.config["betas_for_discretisation"])
        self.loss_function = loss_function_factory(self.config["loss_function"])
        self.loss_function_c = loss_function_factory(self.config["loss_function_c"])
        self.device = choose_device(self.config["device"])

        # Load network
        self._value_network = value_network
        self._value_network = self._value_network.to(self.device)
        self.n_actions = self._value_network.predict.out_features // 2

        self.writer = writer
        if writer:
            self.writer.add_graph(self._value_network,
                                  input_to_model=torch.tensor(np.zeros((1, 1, self._value_network.size_state + 1),
                                                                       dtype=np.float32)).to(self.device))

        self.memory = ReplayMemory(transition_type=TransitionBFTQ, config=self.config)
        self.optimizer = None
        self.batch = 0
        self.epoch = 0
        self.reset()

    def push(self, state, action, reward, next_state, terminal, cost, beta=None):
        """
            Push a transition into the replay memory.
        """
        action = torch.tensor([[action]], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)
        terminal = torch.tensor([terminal], dtype=torch.bool)
        cost = torch.tensor([cost], dtype=torch.float)
        state = torch.tensor([[state]], dtype=torch.float)
        next_state = torch.tensor([[next_state]], dtype=torch.float)

        # Data augmentation for (potentially missing) budget values
        if np.size(self.betas_for_duplication):
            for beta_d in self.betas_for_duplication:
                if beta:  # If the transition already has a beta, augment data by altering it.
                    beta_d = torch.tensor([[[beta_d * beta]]], dtype=torch.float)
                else:  # Otherwise, simply set new betas
                    beta_d = torch.tensor([[[beta_d]]], dtype=torch.float)
                self.memory.push(state, action, reward, next_state, terminal, cost, beta_d)
        else:
            beta = torch.tensor([[[beta]]], dtype=torch.float)
            self.memory.push(state, action, reward, next_state, terminal, cost, beta)

    def run(self):
        """
            Run BFTQ on the batch of transitions in memory.

            We fit a model for the optimal reward-cost state-budget-action values Qr and Qc.
            The BFTQ epoch is repeated until convergence or timeout.
        :return: the obtained value network Qr*, Qc*
        """
        logger.info("Run")
        self.batch += 1
        for self.epoch in range(self.config["epochs"]):
            self._epoch()
        return self._value_network

    def _epoch(self):
        """
            Run a single epoch of BFTQ.

            This is similar to a fitted value iteration:
            1. Bootstrap the targets for Qr, Qc using the Budgeted Bellman Optimality operator
            2. Fit the Qr, Qc model to the targets
        """
        logger.debug("Epoch {}/{}".format(self.epoch + 1, self.config["epochs"]))
        states_betas, actions, rewards, costs, next_states, betas, terminals = self._zip_batch()
        target_r, target_c = self.compute_targets(rewards, costs, next_states, betas, terminals)
        self._fit(states_betas, actions, target_r, target_c)
        plot_values_histograms(self._value_network, (target_r, target_c), states_betas, actions, self.writer, self.epoch, self.batch)

    def _zip_batch(self):
        """
            Convert the batch of transitions to several tensors of states, actions, rewards, etc.
        :return: state-beta, state, action, reward, constraint, next_state, beta, terminal batches
        """
        batch = self.memory.memory
        self.size_batch = len(batch)
        zipped = TransitionBFTQ(*zip(*batch))
        actions = torch.cat(zipped.action).to(self.device)
        rewards = torch.cat(zipped.reward).to(self.device)
        terminals = torch.cat(zipped.terminal).to(self.device)
        costs = torch.cat(zipped.cost).to(self.device)

        betas = torch.cat(zipped.beta).to(self.device)
        states = torch.cat(zipped.state).to(self.device)
        next_states = torch.cat(zipped.next_state).to(self.device)
        states_betas = torch.cat((states, betas), dim=2).to(self.device)

        # Batch normalization
        mean = torch.mean(states_betas, 0).to(self.device)
        std = torch.std(states_betas, 0).to(self.device)
        self._value_network.set_normalization_params(mean, std)

        return states_betas, actions, rewards, costs, next_states, betas, terminals

    def compute_targets(self, rewards, costs, next_states, betas, terminals):
        """
            Compute target values by applying the Budgeted Bellman Optimality operator
        :param rewards: batch of rewards
        :param costs: batch of costs
        :param next_states: batch of next states
        :param betas: batch of budgets
        :param terminals: batch of terminations
        :return: target values
        """
        logger.debug("Compute targets")
        with torch.no_grad():
            next_rewards, next_costs = self.boostrap_next_values(next_states, betas, terminals)
            target_r = rewards + self.config["gamma"] * next_rewards
            target_c = costs + self.config["gamma_c"] * next_costs

            if self.config["clamp_qc"] is not None:
                target_c = torch.clamp(target_c, min=self.config["clamp_qc"][0], max=self.config["clamp_qc"][1])
            torch.cuda.empty_cache()
        return target_r, target_c

    def boostrap_next_values(self, next_states, betas, terminals):
        """
            Boostrap the (Vr, Vc) values at next states by following the greedy policy.

            The model is evaluated for optimal one-step mixtures of actions & budgets that fulfill the cost constraints.

        :param next_states: batch of next states
        :param betas: batch of budgets
        :param terminals: batch of terminations
        :return: Vr and Vc at the next states, following optimal mixtures
        """
        # Initialisation
        next_rewards = torch.zeros(len(next_states), device=self.device)
        next_costs = torch.zeros(len(next_states), device=self.device)
        if self.epoch == 0:
            return next_rewards, next_costs

        # Greedy policy computation pi(a'|s')
        # 1. Select non-final next states
        next_states_nf = next_states[~terminals]
        betas_nf = betas[~terminals]
        # 2. Forward pass of the model Qr, Qc
        q_values = self.compute_next_values(next_states_nf)
        # 3. Compute Pareto-optimal frontiers F of {(Qc, Qr)}_AB at all states
        hulls = self.compute_all_frontiers(q_values, len(next_states_nf))
        # 4. Compute optimal mixture policies satisfying budget constraint: max E[Qr] s.t. E[Qc] < beta
        mixtures = self.compute_all_optimal_mixtures(hulls, betas_nf)

        # Expected value Vr,Vc of the greedy policy at s'
        next_rewards_nf = torch.zeros(len(next_states_nf), device=self.device)
        next_costs_nf = torch.zeros(len(next_states_nf), device=self.device)
        for i, mix in enumerate(mixtures):
            next_rewards_nf[i] = (1 - mix.probability_sup) * mix.inf.qr + mix.probability_sup * mix.sup.qr
            next_costs_nf[i] = (1 - mix.probability_sup) * mix.inf.qc + mix.probability_sup * mix.sup.qc
        next_rewards[~terminals] = next_rewards_nf
        next_costs[~terminals] = next_costs_nf

        torch.cuda.empty_cache()
        return next_rewards, next_costs

    def compute_next_values(self, next_states):
        """
            Compute Q(S, B) with a single forward pass.

            S: set of states
            B: set of budgets (discretised)
        :param next_states: batch of next state
        :return: Q values at next states
        """
        logger.debug("-Forward pass")
        # Compute the cartesian product sb of all next states s with all budgets b
        ss = next_states.squeeze().repeat((1, len(self.betas_for_discretisation))) \
            .view((len(next_states) * len(self.betas_for_discretisation), self._value_network.size_state))
        bb = torch.from_numpy(self.betas_for_discretisation).float().unsqueeze(1).to(device=self.device)
        bb = bb.repeat((len(next_states), 1))
        sb = torch.cat((ss, bb), dim=1).unsqueeze(1)

        # To avoid spikes in memory, we actually split the batch in several minibatches
        batch_sizes = near_split(x=len(sb), num_bins=self.config["split_batches"])
        q_values = []
        for minibatch in range(self.config["split_batches"]):
            mini_batch = sb[sum(batch_sizes[:minibatch]):sum(batch_sizes[:minibatch + 1])]
            q_values.append(self._value_network(mini_batch))
            torch.cuda.empty_cache()
        return torch.cat(q_values).detach().cpu().numpy()

    def compute_all_frontiers(self, q_values, states_count):
        """
            Parallel computing of pareto-optimal frontiers F
        """
        logger.debug("-Compute frontiers")
        n_beta = len(self.betas_for_discretisation)
        hull_params = [(q_values[state * n_beta: (state + 1) * n_beta],
                        self.betas_for_discretisation,
                        self.config["hull_options"],
                        self.config["clamp_qc"])
                       for state in range(states_count)]
        if self.config["processes"] == 1:
            results = [pareto_frontier(*param) for param in hull_params]
        else:
            with Pool(self.config["processes"]) as p:
                results = p.starmap(pareto_frontier, hull_params)
        frontiers, all_points = zip(*results)

        torch.cuda.empty_cache()
        for s in [0, -1]:
            plot_frontier(frontiers[s], all_points[s], self.writer, self.epoch, title="agent/Hull {} batch {}".format(s, self.batch))
        return frontiers

    def compute_all_optimal_mixtures(self, frontiers, betas):
        """
            Parallel computing of optimal mixtures
        """
        logger.debug("-Compute optimal mixtures")
        params = [(frontiers[i], beta.detach().item()) for i, beta in enumerate(betas)]
        if self.config["processes"] == 1:
            optimal_policies = [optimal_mixture(*param) for param in params]
        else:
            with Pool(self.config["processes"]) as p:
                optimal_policies = p.starmap(optimal_mixture, params)
        return optimal_policies

    def _fit(self, states_betas, actions, target_r, target_c):
        """
            Fit a network Q(state, action, beta) = (Qr, Qc) to target values
        :param states_betas: batch of states and betas
        :param actions: batch of actions
        :param target_r: batch of target reward-values
        :param target_c: batch of target cost-values
        :return: the Bellman residual delta between the model and target values
        """
        logger.debug("Fit model")
        # Initial Bellman residual
        with torch.no_grad():
            delta = self._compute_loss(states_betas, actions, target_r, target_c).detach().item()
            torch.cuda.empty_cache()

        # Reset network
        if self.config["reset_network_each_epoch"]:
            self.reset_network()

        # Gradient descent
        losses = []
        for nn_epoch in range(self.config["regression_epochs"]):
            loss = self._gradient_step(states_betas, actions, target_r, target_c)
            losses.append(loss)
        torch.cuda.empty_cache()

        return delta

    def _compute_loss(self, states_betas, actions, target_r, target_c):
        """
            Compute the loss between the model values and target values
        :param states_betas: input state-beta batch
        :param actions: input actions batch
        :param target_r: target qr
        :param target_c: target qc
        :return: the weighted loss for expected rewards and costs
        """
        values = self._value_network(states_betas)
        qr = values.gather(1, actions)
        qc = values.gather(1, actions + self.n_actions)
        loss_qc = self.loss_function_c(qc, target_c.unsqueeze(1))
        loss_qr = self.loss_function(qr, target_r.unsqueeze(1))
        w_r, w_c = self.config["weights_losses"]
        loss = w_c * loss_qc + w_r * loss_qr
        return loss

    def _gradient_step(self, states_betas, actions, target_r, target_c):
        loss = self._compute_loss(states_betas, actions, target_r, target_c)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self._value_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.detach().item()

    def save_network(self, path=None):
        path = Path(path) if path else Path("policy.pt")
        torch.save(self._value_network, path)
        return path

    def load_network(self, path=None):
        path = Path(path) if path else Path("policy.pt")
        self._value_network = torch.load(path, map_location=self.device)
        return self._value_network

    def reset_network(self):
        self._value_network.reset()

    def reset(self, reset_weight=True):
        torch.cuda.empty_cache()
        if reset_weight:
            self.reset_network()
        self.optimizer = optimizer_factory(self.config["optimizer"]["type"],
                                           self._value_network.parameters(),
                                           self.config["optimizer"]["learning_rate"],
                                           self.config["optimizer"]["weight_decay"])
        self.epoch = 0


def parse(value):
    try:
        return eval(value)
    except ValueError:
        return value
