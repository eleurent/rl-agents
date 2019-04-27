import copy
from multiprocessing.pool import Pool

import numpy as np
from gym import logger
import torch
import torch.nn.functional as F
from torch import tensor

from rl_agents.agents.budgeted_ftq.budgeted_utils import TransitionBFTQ, compute_convex_hull_from_values, \
    optimal_mixture
from rl_agents.agents.budgeted_ftq.models import loss_function_factory, optimizer_factory
from rl_agents.agents.utils import ReplayMemory, near_split


class BudgetedFittedQ(object):
    def __init__(self, policy_network, config):
        self.config = config

        # Load configs
        try:
            self.betas_for_duplication = eval(self.config["betas_for_duplication"])
        except TypeError:
            self.betas_for_duplication = self.config["betas_for_duplication"]
        try:
            self.betas_for_discretisation = eval(self.config["betas_for_discretisation"])
        except TypeError:
            self.betas_for_discretisation = self.config["betas_for_discretisation"]
        self.loss_function = loss_function_factory(self.config["loss_function"])
        self.loss_function_c = loss_function_factory(self.config["loss_function_c"])
        self.device = self.config["device"]

        # Load network
        self._value_network = policy_network
        self.n_actions = self._value_network.predict.out_features // 2
        self.size_state = self._value_network.size_state
        self.devices = [self.config["device"]]
        self._value_network = self._value_network.to(self.device)

        self.memory = ReplayMemory(transition_type=TransitionBFTQ)
        self.optimizer = None
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
        if self.betas_for_duplication:
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
        :return: the obtained value network Qr, Qc
        """
        for self.epoch in range(self.config["max_ftq_epochs"]):
            delta = self._epoch()
            if delta < self.config["delta_stop"]:
                break
        return self._value_network

    def _epoch(self):
        """
            Run a single epoch of BFTQ.

            This is similar to a fitted value iteration:
            1. Bootstrap the targets for Qr, Qc using the constrained Bellman Operator
            2. Fit the Qr, Qc model to the targets
        :return: delta, the Bellman residual between the model and target values
        """
        sb_batch, a_batch, r_batch, c_batch, ns_batch, b_batch, t_batch = self._zip_batch()
        target_r, target_c = self.compute_targets(r_batch, c_batch, ns_batch, b_batch, t_batch)
        return self._fit(sb_batch, a_batch, target_r, target_c)

    def _zip_batch(self):
        """
            Convert the batch of transitions to several tensors of states, actions, rewards, etc.
        :return: state-beta, state, action, reward, constraint, next_state, beta, terminal batches
        """
        batch = self.memory.memory
        self.size_batch = len(batch)
        zipped = TransitionBFTQ(*zip(*batch))
        action_batch = torch.cat(zipped.action).to(self.device)
        reward_batch = torch.cat(zipped.reward).to(self.device)
        terminal_batch = torch.cat(zipped.terminal).to(self.device)
        constraint_batch = torch.cat(zipped.constraint).to(self.device)

        beta_batch = torch.cat(zipped.beta).to(self.device)
        state_batch = torch.cat(zipped.state).to(self.device)
        next_state_batch = torch.cat(zipped.next_state).to(self.device)
        state_beta_batch = torch.cat((state_batch, beta_batch), dim=2).to(self.device)

        # Batch normalization
        mean = torch.mean(state_beta_batch, 0).to(self.device)
        std = torch.std(state_beta_batch, 0).to(self.device)
        self._value_network.set_normalization_params(mean, std)

        return state_beta_batch, action_batch, reward_batch, constraint_batch, next_state_batch, beta_batch, \
            terminal_batch

    def compute_targets(self, r_batch, c_batch, ns_batch, b_batch, t_batch):
        """
            Compute target values by applying constrained Bellman operator
        :param r_batch: batch of rewards
        :param c_batch: batch of costs
        :param ns_batch: batch of next states
        :param b_batch: batch of budgets
        :param t_batch: batch of terminations
        :return: target values
        """
        with torch.no_grad():
            ns_r, ns_c = self.constrained_next_values(ns_batch, b_batch, t_batch)
            target_r = r_batch + self.config["gamma"] * ns_r
            target_c = c_batch + self.config["gamma_c"] * ns_c

            if self.config["clamp_Qc"] is not None:
                logger.info("Clamp target costs")
                target_c = torch.clamp(target_c, min=self.config["clamp_Qc"][0], max=self.config["clamp_Qc"][1])
            torch.cuda.empty_cache()
        return target_r, target_c

    def constrained_next_values(self, ns_batch, b_batch, t_batch):
        """
            Boostrap the Qr, Qc values according to the current model and under the cost constraints.

            The model is evaluated for the optimal mixtures of actions/budgets fulfilling the cost constraints.

        :param ns_batch: batch of next states
        :param b_batch: batch of budgets
        :param t_batch: batch of terminations
        :return: Qr and Qc at the next states
        """
        # Initialisation
        next_state_rewards = torch.zeros(len(ns_batch), device=self.device)
        next_state_costs = torch.zeros(len(ns_batch), device=self.device)
        if self.epoch > 0:
            return next_state_rewards, next_state_costs

        # Select non-final next states
        ns_batch_nf = ns_batch[1 - t_batch]
        b_batch_nf = b_batch[1 - t_batch]

        # Forward pass of the model Qr, Qc
        q_values = self.compute_next_values(ns_batch_nf)

        # Compute hulls H
        hulls = self.compute_all_hulls(q_values, len(ns_batch_nf))

        # Compute optimal mixture policies satisfying budgets beta
        policies = self.compute_optimal_policies(hulls, b_batch_nf)

        next_state_beta_not_terminal = torch.zeros((len(where_not_terminal_ns) * 2, 1, self.size_state + 1),
                                                   device=self.device)
        h_batch_not_terminal = h_batch[where_not_terminal_ns]
        b_bath_not_terminal = b_batch[where_not_terminal_ns]
        status = {"regular": 0, "not_solvable": 0, "too_much_budget": 0, "exact": 0}
        for i_ns_nt, (next_state, policy) in enumerate(zip(ns_batch_nf, policies)):
            mixture, stat = policy
            status[stat] += 1
            ns_beta_minus = torch.cat((next_state, torch.tensor([[mixture.budget_inf]], device=self.device)), dim=1)
            ns_beta_plus = torch.cat((next_state, torch.tensor([[mixture.budget_sup]], device=self.device)), dim=1)
            next_state_beta_not_terminal[i_ns_nt * 2 + 0][0] = ns_beta_minus
            next_state_beta_not_terminal[i_ns_nt * 2 + 1][0] = ns_beta_plus

        ##############################################################
        # Forwarding to compute the Q function in s' #################
        ##############################################################

        self.info("Q next")
        self.track_memory("Q_next")
        Q_next_state_not_terminal = self._value_network(next_state_beta_not_terminal)
        Q_next_state_not_terminal = Q_next_state_not_terminal.detach()
        self.track_memory("Q_next (end)")
        self.info("Q next end")
        torch.cuda.empty_cache()

        ###########################################
        ############  bootstraping   ##############
        ###########################################

        self.info("computing next values ...")
        self.track_memory("compute_next_values")

        warning_qc_negatif = 0.
        offset_qc_negatif = 0.
        warning_qc__negatif = 0.
        offset_qc__negatif = 0.
        next_state_c_neg = 0.
        offset_next_state_c_neg = 0.

        next_state_rewards_not_terminal = torch.zeros(len(where_not_terminal_ns), device=self.device)
        next_state_constraints_not_terminal = torch.zeros(len(where_not_terminal_ns), device=self.device)

        for i_ns_nt in range(len(where_not_terminal_ns)):
            mixture = opts_and_statuses[i_ns_nt][0]
            qr_ = Q_next_state_not_terminal[i_ns_nt * 2][mixture.action_inf]
            qr = Q_next_state_not_terminal[i_ns_nt * 2 + 1][mixture.action_sup]
            qc_ = Q_next_state_not_terminal[i_ns_nt * 2][self.n_actions + mixture.action_inf]
            qc = Q_next_state_not_terminal[i_ns_nt * 2 + 1][self.n_actions + mixture.action_sup]

            if qc < 0.:
                warning_qc_negatif += 1.
                offset_qc_negatif = qc

            if qc_ < 0.:
                warning_qc__negatif += 1.
                offset_qc__negatif = qc_

            next_state_rewards_not_terminal[i_ns_nt] = mixture.proba_inf * qr_ + mixture.proba_sup * qr
            next_state_constraints_not_terminal[i_ns_nt] = mixture.proba_inf * qc_ + mixture.proba_sup * qc

            if next_state_constraints_not_terminal[i_ns_nt] < 0:
                next_state_c_neg += 1.
                offset_next_state_c_neg = next_state_constraints_not_terminal[i_ns_nt]

        next_state_rewards[where_not_terminal_ns] = next_state_rewards_not_terminal
        next_state_costs[where_not_terminal_ns] = next_state_constraints_not_terminal

        if logger.getEffectiveLevel() <= logging.DEBUG:
            self.info("printing some graphs in next_values ...")
            self.info("\n[compute_next_values] Q(s') sur le batch")
            create_Q_histograms("Qr(s')_e={}".format(self.epoch),
                                values=next_state_rewards.cpu().numpy().flatten(),
                                path=self.workspace / "histogram",
                                labels=["next value"])
            create_Q_histograms("Qc(s')_e={}".format(self.epoch),
                                values=next_state_costs.cpu().numpy().flatten(),
                                path=self.workspace / "histogram",
                                labels=["next value"])
            self.info("printing some graphs in next_values ... done")

        mean_qc_neg = 0 if warning_qc_negatif == 0 else offset_qc_negatif / warning_qc_negatif
        mean_qc__neg = 0 if warning_qc__negatif == 0 else offset_qc__negatif / warning_qc__negatif
        mean_ns_neg = 0 if next_state_c_neg == 0 else offset_next_state_c_neg / next_state_c_neg
        self.info("qc < 0 percentage {:.2f}% with a mean offset of {:.4f}".format(
            warning_qc_negatif / len(where_not_terminal_ns) * 100., mean_qc_neg))
        self.info("qc_ < 0 percentage {:.2f}% with a mean offset of {:.4f}".format(
            warning_qc__negatif / len(where_not_terminal_ns) * 100., mean_qc__neg))
        self.info("next_state_costs < 0 percentage {:.2f}% with a mean offset of {:.4f}".format(
            next_state_c_neg / len(where_not_terminal_ns) * 100., mean_ns_neg))
        self.info("compute next values ... end")
        torch.cuda.empty_cache()
        self.track_memory("compute_next_values (end)")

        return next_state_rewards, next_state_costs

    def compute_next_values(self, ns_batch):
        """
            Compute Q(s, beta) with a single forward pass
        :param ns_batch: batch of next state
        :return: next values
        """
        # Compute the cartesian product sb of all next states s with all budgets b
        ss = ns_batch \
            .squeeze() \
            .repeat((1, len(self.betas_for_discretisation))) \
            .reshape(-1) \
            .reshape((len(ns_batch) * len(self.betas_for_discretisation), self.size_state))
        bb = torch.from_numpy(self.betas_for_discretisation).float().unsqueeze(1).to(device=self.device)
        bb = bb.repeat((len(ns_batch), 1))
        sb = torch.cat((ss, bb), dim=1).unsqueeze(1)

        # To avoid spikes in memory, we actually split the batch in several minibatches
        batch_sizes = near_split(x=len(sb), num_bins=self.config["split_batches"])
        q_values = []
        for i in range(self.config["split_batches"]):
            mini_batch = sb[sum(batch_sizes[:i]):sum(batch_sizes[:i + 1])]
            q_values.append(self._value_network(mini_batch))
            torch.cuda.empty_cache()
        return torch.cat(q_values).detach().cpu().numpy()

    def compute_all_hulls(self, q_values, n_states):
        n_beta = len(self.betas_for_discretisation)
        hull_params = [(q_values[state * n_beta: (state + 1) * n_beta],
                        self.betas_for_discretisation,
                        self.config["hull_options"],
                        self.config["clamp_Qc"])
                       for state in range(n_states)]
        with Pool(self.config["cpu_processes"]) as p:
            results = p.starmap(compute_convex_hull_from_values, hull_params)
            hulls, _, _, _ = zip(*results)
        torch.cuda.empty_cache()
        return hulls

    def compute_optimal_policies(self, b_batch, hulls):
        with Pool(self.config["cpu_processes"]) as p:
            optimal_policies = p.starmap(optimal_mixture,
                                         [(hulls[i], beta.detach().item()) for i, beta in enumerate(b_batch)])
        return optimal_policies

    def _fit(self, sb_batch, a_batch, label_r, label_c):
        sb_batch = sb_batch.to(self.device)
        a_batch = a_batch.to(self.device)
        label_r = label_r.to(self.device)
        label_c = label_c.to(self.device)
        self.info("optimize model ...")
        self.track_memory("delta")
        with torch.no_grad():
            # no need gradient just for computing delta ....
            delta = self._compute_loss(sb_batch, a_batch, label_r, label_c, with_weight=False).detach().item()
            torch.cuda.empty_cache()
        self.track_memory("delta (end)")
        self.info("reset neural network ? {}".format(self.RESET_POLICY_NETWORK_EACH_FTQ_EPOCH))
        if self.RESET_POLICY_NETWORK_EACH_FTQ_EPOCH:
            self.reset_network()
        stop = False
        nn_epoch = 0
        losses = []
        last_loss = np.inf
        self.info("gradient descent ...")
        self.track_memory("GD")

        while not stop:
            loss = self._gradient_step(sb_batch, a_batch, label_r, label_c)
            losses.append(loss)
            if (min(last_loss, loss) / max(last_loss, loss) < 0.5 or nn_epoch in [0, 1, 2, 3]):
                self.info("[epoch_nn={:03}] loss={:.4f}".format(nn_epoch, loss))
            last_loss = loss
            cvg = loss < self.stop_loss_value
            if cvg:
                self.info("[epoch_nn={:03}] early stopping [loss={}]".format(nn_epoch, loss))
            nn_epoch += 1
            stop = nn_epoch > self._MAX_NN_EPOCH or cvg
        self.track_memory("GD (end)")
        if not cvg:
            for i in range(3):
                self.info("[epoch_nn={:03}] loss={:.4f}".format(nn_epoch - 3 + i, losses[-3 + i]))
        self.info("gradient descent ... end")
        torch.cuda.empty_cache()
        self.info("optimize model ... done")
        self.track_memory("optimize model (end)")
        return delta

    def _compute_loss(self, sb_batch, a_batch, label_r, label_c, with_weight=True):
        output = self._value_network(sb_batch)
        state_action_rewards = output.gather(1, a_batch)
        state_action_constraints = output.gather(1, a_batch + self.n_actions)
        loss_Qc = self.loss_function_c(state_action_constraints, label_c.unsqueeze(1))
        loss_Qr = self.loss_function(state_action_rewards, label_r.unsqueeze(1))
        w_r, w_c = self.weights_losses
        if with_weight:
            loss = w_c * loss_Qc + w_r * loss_Qr
        else:
            loss = loss_Qc + loss_Qr
        return loss

    def _gradient_step(self, sb_batch, a_batch, label_r, label_c):
        loss = self._compute_loss(sb_batch, a_batch, label_r, label_c)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self._value_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.detach().item()

    def save_policy(self, policy_path=None):
        if policy_path is None:
            policy_path = self.workspace / "policy.pt"
        self.info("saving bftq policy at {}".format(policy_path))
        torch.save(self._value_network, policy_path)
        return policy_path

    def _is_terminal_state(self, state):
        isnan = torch.sum(torch.isnan(state)) == self.size_state
        return isnan

    def empty_cache(self):
        torch.cuda.empty_cache()

    def reset_network(self):
        if self.use_data_parallel:
            self._value_network.module.reset()
        else:
            self._value_network.reset()

    def reset(self, reset_weight=True):
        torch.cuda.empty_cache()
        if reset_weight:
            self.reset_network()
        self.optimizer = optimizer_factory(self.optimizer_type,
                                           self._value_network.parameters(),
                                           self.learning_rate,
                                           self.weight_decay)
        self.epoch = None

    def format_memory(self, memoire):
        for _ in range(len(self.devices) - len(memoire)):
            memoire.append(0)
        accolades = "".join(["{:05} " for _ in range(len(self.devices))])
        accolades = accolades[:-1]
        format = "[m=" + accolades + "]"
        return format.format(*memoire)

    def get_current_memory(self):
        memory = get_memory_for_pid(os.getpid())

        # self.memory_tracking.append(sum)
        return memory

    def track_memory(self, id):
        sum = 0
        for mem in self.get_current_memory():
            sum += mem
        self.memory_tracking.append([id, sum])

    def info(self, message):
        memoire = self.get_current_memory()

        if self.epoch is not None:
            logger.info("[e={:02}]{} {}".format(self.epoch, self.format_memory(memoire), message))
        else:
            logger.info("{} {} ".format(self.format_memory(memoire), message))
