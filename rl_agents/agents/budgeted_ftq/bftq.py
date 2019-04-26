import copy
from multiprocessing.pool import Pool

import numpy as np
from gym import logger
import torch
import torch.nn.functional as F
from torch import tensor

from rl_agents.agents.budgeted_ftq.budgeted_utils import TransitionBFTQ, convex_hull_qsb
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
        self._policy_network = policy_network
        self.n_actions = self._policy_network.predict.out_features // 2
        self.size_state = self._policy_network.size_state
        self.devices = [self.config["device"]]
        self._policy_network = self._policy_network.to(self.device)

        self.memory = ReplayMemory(transition_type=TransitionBFTQ)
        self.optimizer = None
        self.ftq_epoch = 0
        self.reset()

    def push(self, state, action, reward, next_state, terminal, constraint, beta=None):
        action = torch.tensor([[action]], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)
        terminal = torch.tensor([terminal], dtype=torch.bool)
        constraint = torch.tensor([constraint], dtype=torch.float)
        state = torch.tensor([[state]], dtype=torch.float)
        next_state = torch.tensor([[next_state]], dtype=torch.float)
        if self.betas_for_duplication:
            for beta_d in self.betas_for_duplication:
                if beta:  # If the transition already has a beta, augment data by altering it.
                    beta_d = torch.tensor([[[beta_d * beta]]], dtype=torch.float)
                else:  # Otherwise, simply set new betas
                    beta_d = torch.tensor([[[beta_d]]], dtype=torch.float)
                self.memory.push(state, action, reward, next_state, terminal, constraint, beta_d)
        else:
            beta = torch.tensor([[[beta]]], dtype=torch.float)
            self.memory.push(state, action, reward, next_state, terminal, constraint, beta)

    def _construct_batch(self):
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
        self._policy_network.set_normalization_params(mean, std)

        return state_beta_batch, state_batch, action_batch, reward_batch, constraint_batch, next_state_batch, \
               beta_batch, terminal_batch

    def fit(self, transitions):
        self.reset_network()
        batches = self._construct_batch(transitions)
        delta = np.inf
        self.ftq_epoch = 0
        while self.ftq_epoch < self.config["max_ftq_epochs"] and delta > self.config["delta_stop"]:
            _ = self._ftq_epoch(batches)
            self.info("delta = {}".format(self.delta))
            self.ftq_epoch += 1
        return self._policy_network

    def _ftq_epoch(self, batches):
        sb_batch, s_batch, a_batch, r_batch, c_batch, ns_batch, b_batch, t_batch = batches
        with torch.no_grad():
            if self.ftq_epoch > 0:
                ns_r, ns_c = self.compute_next_values(ns_batch, b_batch, t_batch)
            else:
                ns_r, ns_c = torch.zeros(self.size_batch, device=self.device), \
                             torch.zeros(self.size_batch, device=self.device)

            target_r = r_batch + self.config["gamma"] * ns_r
            target_c = c_batch + self.config["gamma_c"] * ns_c

            if self.config["clamp_Qc"] is not None:
                logger.info("Clamp target constraints")
                target_c = torch.clamp(target_c, min=self.config["clamp_Qc"][0], max=self.config["clamp_Qc"][1])
            self.empty_cache()
        losses = self._optimize_model(sb_batch, a_batch, target_r, target_c)
        self.empty_cache()
        return losses

    def compute_next_values(self, ns_batch, b_batch, t_batch):
        next_state_rewards = torch.zeros(self.size_batch, device=self.device)
        next_state_constraints = torch.zeros(self.size_batch, device=self.device)

        with torch.no_grad():
            ################################################
            # computing all unique hulls (no terminal next_state and no repetition among next state)
            ################################################
            self.track_memory("compute hulls")
            self.info("computing hulls ...")

            # gather all s,beta to compute in one foward pass (may increase the spike of memory used temporary)
            # but essential for multiprocessing hull computing bellow

            ns_batch_non_final = ns_batch[1 - t_batch]

            self.info("There are {} hulls to compute !".format(len(ns_batch_non_final)))

            """ 
            sb = "duplicating each next states with each beta"
            if we have 2 states s0 and s1 and 2 betas 0 and 1. What do we want for sb is :
            (s0,0)
            (s0,1)
            (s1,0)
            (s1,1)
            """
            ss = ns_batch_non_final \
                .squeeze() \
                .repeat((1, len(self.betas_for_discretisation))) \
                .reshape(-1) \
                .reshape((len(ns_batch_non_final) * len(self.betas_for_discretisation), self.size_state))
            bb = torch.from_numpy(self.betas_for_discretisation).float().unsqueeze(1).to(device=self.device)
            bb = bb.repeat((len(ns_batch_non_final), 1))
            sb = torch.cat((ss, bb), dim=1)
            sb = sb.unsqueeze(1)

            batch_sizes = near_split(x=len(sb), num_bins=self.config["split_batches"])
            q_values = []
            self.info("Splitting states batch in minibatches to avoid out of memory")
            self.info("Batch sizes : {}".format(batch_sizes))
            offset = 0
            for i in range(self.config["split_batches"]):
                self.info("mini batch {}".format(i))
                self.track_memory("mini_batch={}".format(i))
                mini_batch = sb[offset:offset + batch_sizes[i]]
                offset += batch_sizes[i]
                q_values.append(self._policy_network(mini_batch))
                torch.cuda.empty_cache()
            q_values = torch.cat(q_values)

            q_values = q_values.detach().cpu().numpy()

            hulls_to_compute = [(
                    q_values[(state * len(self.betas_for_discretisation)):
                             (state + 1) * len(self.betas_for_discretisation)],
                    self.betas_for_discretisation,
                    self.config["hull_options"],
                    self.config["clamp_Qc"])
                 for state in range(len(ns_batch_non_final))
            ]
            with Pool(self.config["cpu_processes"]) as p:
                # p.map return ordered fashion, so we're cool
                results = p.starmap(convex_hull_qsb, hulls_to_compute)
                hulls_for_ns_batch_unique, colinearities, true_colinearities, exceptions = zip(*results)

            self.info("exceptions : {:.4f} %".format(
                np.sum(np.array(exceptions)) / len(hulls_for_ns_batch_unique) * 100.))
            self.info("true_colinearities : {:.4f} %".format(
                np.sum(np.array(true_colinearities)) / len(hulls_for_ns_batch_unique) * 100.))
            self.info("colinearities : {:.4f} %".format(
                np.sum(np.array(colinearities)) / len(hulls_for_ns_batch_unique) * 100.))
            self.info("#next_states : {}".format(len(ns_batch)))
            self.info("#non terminal next_states : {}".format(len(where_not_terminal_ns)))
            self.info("#hulls actually computed : {}".format(len(hulls_for_ns_batch_unique)))
            self.info("computing hulls [DONE] ")
            self.empty_cache()
            self.track_memory("compute hulls (end)")

            #####################################
            # for each next_state
            # computing optimal distribution among 2 actions,
            # with respectively 2 budget
            # given the hull of the next_state and the beta (in current state).
            #####################################

            self.track_memory("compute_opts")
            self.info("computing ops ... ")

            ##############################################################
            # we build couples (s',beta\top) and (s',beta\bot)
            # in order to compute Q(s')
            ##############################################################
            next_state_beta_not_terminal = torch.zeros((len(where_not_terminal_ns) * 2, 1, self.size_state + 1),
                                                       device=self.device)
            ns_batch_not_terminal = ns_batch[where_not_terminal_ns]
            h_batch_not_terminal = h_batch[where_not_terminal_ns]
            b_bath_not_terminal = b_batch[where_not_terminal_ns]

            args = [(beta.detach().item(), hulls_for_ns_batch_unique[hull_id], {})
                    for hull_id, beta in zip(h_batch_not_terminal, b_bath_not_terminal)]

            self.info("computing optimal_pia_pib in parralle ...")
            with Pool(self.config["cpu_processes"]) as p:
                opts_and_statuses = p.map(optimal_pia_pib_parralle, args)
            self.info("computing optimal_pia_pib in parralle ... done")

            self.info("computing opts ... end")
            self.track_memory("compute_opts (end)")
            status = {"regular": 0, "not_solvable": 0, "too_much_budget": 0, "exact": 0}

            for i_ns_nt, (ns_not_terminal, opt_and_status) in enumerate(
                    zip(ns_batch_not_terminal, opts_and_statuses)):
                opt, stat = opt_and_status
                status[stat] += 1
                ns_beta_moins = torch.cat((ns_not_terminal, torch.tensor([[opt.budget_inf]], device=self.device)),
                                          dim=1)
                ns_beta_plus = torch.cat((ns_not_terminal, torch.tensor([[opt.budget_sup]], device=self.device)),
                                         dim=1)
                next_state_beta_not_terminal[i_ns_nt * 2 + 0][0] = ns_beta_moins
                next_state_beta_not_terminal[i_ns_nt * 2 + 1][0] = ns_beta_plus

            ##############################################################
            # Forwarding to compute the Q function in s' #################
            ##############################################################

            self.info("Q next")
            self.track_memory("Q_next")
            Q_next_state_not_terminal = self._policy_network(next_state_beta_not_terminal)
            Q_next_state_not_terminal = Q_next_state_not_terminal.detach()
            self.track_memory("Q_next (end)")
            self.info("Q next end")
            self.empty_cache()

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
                opt = opts_and_statuses[i_ns_nt][0]
                qr_ = Q_next_state_not_terminal[i_ns_nt * 2][opt.id_action_inf]
                qr = Q_next_state_not_terminal[i_ns_nt * 2 + 1][opt.id_action_sup]
                qc_ = Q_next_state_not_terminal[i_ns_nt * 2][self.n_actions + opt.id_action_inf]
                qc = Q_next_state_not_terminal[i_ns_nt * 2 + 1][self.n_actions + opt.id_action_sup]

                if qc < 0.:
                    warning_qc_negatif += 1.
                    offset_qc_negatif = qc

                if qc_ < 0.:
                    warning_qc__negatif += 1.
                    offset_qc__negatif = qc_

                next_state_rewards_not_terminal[i_ns_nt] = opt.proba_inf * qr_ + opt.proba_sup * qr
                next_state_constraints_not_terminal[i_ns_nt] = opt.proba_inf * qc_ + opt.proba_sup * qc

                if next_state_constraints_not_terminal[i_ns_nt] < 0:
                    next_state_c_neg += 1.
                    offset_next_state_c_neg = next_state_constraints_not_terminal[i_ns_nt]

            next_state_rewards[where_not_terminal_ns] = next_state_rewards_not_terminal
            next_state_constraints[where_not_terminal_ns] = next_state_constraints_not_terminal

            if logger.getEffectiveLevel() <= logging.DEBUG:
                self.info("printing some graphs in next_values ...")
                self.info("\n[compute_next_values] Q(s') sur le batch")
                create_Q_histograms("Qr(s')_e={}".format(self.ftq_epoch),
                                    values=next_state_rewards.cpu().numpy().flatten(),
                                    path=self.workspace / "histogram",
                                    labels=["next value"])
                create_Q_histograms("Qc(s')_e={}".format(self.ftq_epoch),
                                    values=next_state_constraints.cpu().numpy().flatten(),
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
            self.info("next_state_constraints < 0 percentage {:.2f}% with a mean offset of {:.4f}".format(
                next_state_c_neg / len(where_not_terminal_ns) * 100., mean_ns_neg))
            self.info("compute next values ... end")
            self.empty_cache()
            self.track_memory("compute_next_values (end)")

        return next_state_rewards, next_state_constraints

    def _optimize_model(self, sb_batch, a_batch, label_r, label_c):
        sb_batch = sb_batch.to(self.device)
        a_batch = a_batch.to(self.device)
        label_r = label_r.to(self.device)
        label_c = label_c.to(self.device)
        self.info("optimize model ...")
        self.track_memory("delta")
        with torch.no_grad():
            self.info("computing delta ...")
            # no need gradient just for computing delta ....
            if self.use_data_loader:
                logger.warning("--- We are not computing delta ---")
                self.delta = np.inf
            else:
                self.delta = self._compute_loss(sb_batch, a_batch, label_r, label_c, with_weight=False).detach().item()
            self.info("computing delta ... done")
            self.empty_cache()
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
        # del label_r, label_c
        self.empty_cache()
        self.info("optimize model ... done")
        self.track_memory("optimize model (end)")
        return losses

    def _compute_loss(self, sb_batch, a_batch, label_r, label_c, with_weight=True):
        output = self._policy_network(sb_batch)
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
        for param in self._policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.detach().item()

    def save_policy(self, policy_path=None):
        if policy_path is None:
            policy_path = self.workspace / "policy.pt"
        self.info("saving bftq policy at {}".format(policy_path))
        torch.save(self._policy_network, policy_path)
        return policy_path

    def _is_terminal_state(self, state):
        isnan = torch.sum(torch.isnan(state)) == self.size_state
        return isnan

    def empty_cache(self):
        torch.cuda.empty_cache()

    def reset_network(self):
        if self.use_data_parallel:
            self._policy_network.module.reset()
        else:
            self._policy_network.reset()

    def reset(self, reset_weight=True):
        torch.cuda.empty_cache()
        if reset_weight:
            self.reset_network()
        self.optimizer = optimizer_factory(self.optimizer_type,
                                           self._policy_network.parameters(),
                                           self.learning_rate,
                                           self.weight_decay)
        self.ftq_epoch = None

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

        if self.ftq_epoch is not None:
            logger.info("[e={:02}]{} {}".format(self.ftq_epoch, self.format_memory(memoire), message))
        else:
            logger.info("{} {} ".format(self.format_memory(memoire), message))
