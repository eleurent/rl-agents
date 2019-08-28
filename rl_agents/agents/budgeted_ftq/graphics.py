import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import re


class BFTQGraphics(object):
    @classmethod
    def display(cls, agent, surface):
        import pygame
        mixture, hull = agent.exploration_policy.pi_greedy.greedy_policy(agent.previous_state, agent.previous_beta)
        surf_size = surface.get_size()
        image_str, size = plot_frontier(*hull, None, None, "", beta=agent.previous_beta, mixture=mixture,
                                        figsize=(surf_size[0]/100, surf_size[1]/100), verbose=False,
                                        clamp_qc=agent.config["clamp_qc"])
        surf = pygame.image.fromstring(image_str, size, "RGB")
        surface.blit(surf, (0, 0))


def plot_values_histograms(value_network, targets, states_betas, actions, writer, epoch, batch):
    with torch.no_grad():
        values = value_network(states_betas)
    n_actions = value_network.predict.out_features // 2
    targets_r, targets_c = targets
    # Histograms of values of observed transitions
    plot_histograms(title="agent/Qr (observed transitions) batch {}".format(batch), writer=writer, epoch=epoch, labels=["target", "prediction"],
                    values=[targets_r.cpu().numpy(), values.gather(1, actions).cpu().numpy()])
    plot_histograms(title="agent/Qc (observed transitions) batch {}".format(batch), writer=writer, epoch=epoch, labels=["target", "prediction"],
                    values=[targets_c.cpu().numpy(), values.gather(1, actions + n_actions).cpu().numpy()])
    # Histograms of values of all possible actions
    plot_histograms(title="agent/Qr (all actions) batch {}".format(batch), writer=writer, epoch=epoch, labels=map(str, range(n_actions)),
                    values=values[:, :n_actions].cpu().numpy().transpose())
    plot_histograms(title="agent/Qc (all actions) batch {}".format(batch), writer=writer, epoch=epoch, labels=map(str, range(n_actions)),
                    values=values[:, -n_actions:].cpu().numpy().transpose())


def plot_histograms(title, writer, epoch, labels, values):
    fig = plt.figure()
    for value, label in zip(values, labels):
        sns.distplot(value, label=label)
    plt.title(title)
    plt.legend(loc='upper right')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = np.rollaxis(data.reshape(fig.canvas.get_width_height()[::-1] + (3,)), 2, 0)
    writer.add_image(clean_tag(title), data, epoch)
    plt.close()


def plot_frontier(frontier, all_points, writer=None, epoch=0, title="", beta=None, mixture=None, figsize=(8, 6),
                  verbose=True, clamp_qc=None):
    """
        Plot the hull of all Qc, Qr points for different (action, budget).

        If a threshold beta and corresponding mixture is provided, plot them.
    :param frontier: points of the Pareto frontier
    :param all_points: all points (Qc, Qr)
    :param SummaryWriter writer: will log the image to tensorboard if not None
    :param epoch: timestep for tensorboard log
    :param title: figure title
    :param beta: a budget threshold used at decision time
    :param mixture: the optimal mixture corresponding to this budget beta
    :param figsize: figure size, inches
    :param verbose: should the legend be displayed
    :param clamp_qc: if qc is clamped, use these values at x axis limits
    :return: the string description of the image, and its size
    """
    # Figure creation
    dfa, dfh = pd.DataFrame(all_points), pd.DataFrame(frontier)
    fig = plt.figure(figsize=figsize, tight_layout=True)
    sns.scatterplot(data=dfa, x="qc", y="qr", hue="action", legend="full")
    sns.lineplot(data=dfh, x="qc", y="qr", marker="x", label="hull")
    if clamp_qc:  # known limits
        plt.xlim(clamp_qc[0]-0.1, clamp_qc[1]+0.1)
    if beta is not None:
        plt.axvline(x=beta)
    if mixture:
        sns.lineplot(x=[mixture.inf.qc, mixture.sup.qc], y=[mixture.inf.qr, mixture.sup.qr],
                     color="red", marker="o")
    plt.title(title)
    leg = plt.legend(loc='upper right')
    if not verbose:
        leg.remove()
        plt.xlabel('')
        plt.ylabel('')

    # Figure export
    fig.canvas.draw()
    data_str = fig.canvas.tostring_rgb()
    if writer:
        data = np.fromstring(data_str, dtype=np.uint8, sep='')
        data = np.rollaxis(data.reshape(fig.canvas.get_width_height()[::-1] + (3,)), 2, 0)
        writer.add_image(clean_tag(title), data, epoch)
    plt.close()
    return data_str, fig.canvas.get_width_height()


def clean_tag(tag):
    """
        Clean image tags before logging to tensorboard
    """
    invalid_characters = re.compile(r'[^-/\w\.]')
    return invalid_characters.sub('_', tag)
