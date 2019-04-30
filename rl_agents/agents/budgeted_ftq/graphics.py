import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_values_histograms(value_network, targets, states_betas, actions, writer, epoch, batch):
    with torch.no_grad():
        values = value_network(states_betas)
    n_actions = value_network.predict.out_features // 2
    targets_r, targets_c = targets
    # Histograms of values of observed transitions
    plot_histograms(title="Qr (observed transitions) batch {}".format(batch), writer=writer, epoch=epoch, labels=["target", "prediction"],
                    values=[targets_r.cpu().numpy(), values.gather(1, actions).cpu().numpy()])
    plot_histograms(title="Qc (observed transitions) batch {}".format(batch), writer=writer, epoch=epoch, labels=["target", "prediction"],
                    values=[targets_c.cpu().numpy(), values.gather(1, actions + n_actions).cpu().numpy()])
    # Histograms of values of all possible actions
    plot_histograms(title="Qr (all actions) batch {}".format(batch), writer=writer, epoch=epoch, labels=map(str, range(n_actions)),
                    values=values[:, :n_actions].cpu().numpy())
    plot_histograms(title="Qc (all actions) batch {}".format(batch), writer=writer, epoch=epoch, labels=map(str, range(n_actions)),
                    values=values[:, -n_actions:].cpu().numpy())


def plot_histograms(title, writer, epoch, labels, values):
    fig = plt.figure()
    for value, label in zip(values, labels):
        sns.distplot(value, label=label)
    plt.title(title)
    plt.legend(loc='upper right')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = np.rollaxis(data.reshape(fig.canvas.get_width_height()[::-1] + (3,)), 2, 0)
    writer.add_image(title, data, epoch)
    plt.close()


def scatter_state_values(top_points, points, all_points, writer, batch, epoch):
    title = "Values Hull batch {}".format(batch)
    fig = plt.figure()
    sns.scatterplot(x=[p.qc for p in all_points], y=[p.qr for p in all_points], label="all")
    sns.scatterplot(x=[p.qc for p in points], y=[p.qr for p in points], label="undominated")
    sns.lineplot(x=[p.qc for p in top_points], y=[p.qr for p in top_points], marker="x", label="hull")
    plt.title(title)
    plt.legend(loc='upper right')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = np.rollaxis(data.reshape(fig.canvas.get_width_height()[::-1] + (3,)), 2, 0)
    writer.add_image(title, data, epoch)
    plt.close()
