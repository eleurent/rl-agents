from collections import defaultdict
import gym
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import logging

from rl_agents.agents.tree_search.graphics import TreePlot
from rl_agents.agents.common.factory import agent_factory, load_environment
from rl_agents.trainer.logger import configure
from utils.envs import GridEnv

sns.set()
sns.set(font_scale=1.5, rc={'text.usetex': True})

logger = logging.getLogger(__name__)

out = Path("out/planners")
gamma = 0.95

envs = {
    "highway": Path("configs") / "HighwayEnv" / "env.json",
    "bandit": Path("configs") / "FiniteMDPEnv" / "env_bandit.json",
    "env_loop": Path("configs") / "FiniteMDPEnv" / "env_loop.json",
    "env_garnet": Path("configs") / "FiniteMDPEnv" / "env_garnet.json",
    "gridenv": Path("configs") / "DummyEnv" / "gridenv.json",
    "gridenv_stoch": Path("configs") / "DummyEnv" / "gridenv_stoch.json",
    "dynamics": Path("configs") / "DummyEnv" / "dynamics.json",
}

agents = {
    "olop": {
        "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
        "gamma": gamma,
        "upper_bound": {
            "type": "hoeffding",
            "c": 4
        },
        "lazy_tree_construction": True,
        "continuation_type": "uniform",
    },
    "KL-OLOP": {
        "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
        "gamma": gamma,
        "upper_bound": {
            "type": "kullback-leibler",
            "threshold": "1*np.log(time)"
        },
        "lazy_tree_construction": True,
        "continuation_type": "uniform",
    },
    "laplace": {
        "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
        "gamma": gamma,
        "upper_bound": {
            "type": "laplace",
            "c": 2
        },
        "lazy_tree_construction": True,
        "continuation_type": "uniform",
    },
    "OPD": {
        "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
        "gamma": gamma,
    },
    "GBOP-T": {
        "__class__": "<class 'rl_agents.agents.tree_search.state_aware.StateAwarePlannerAgent'>",
        "gamma": gamma,
        "backup_aggregated_nodes": True,
        "prune_suboptimal_leaves": True,
        "accuracy": 1e-3
    },
    "GBOP-D": {
        "__class__": "<class 'rl_agents.agents.tree_search.graph_based.GraphBasedPlannerAgent'>",
        "gamma": gamma,
    },
    "GBOP": {
        "__class__": "<class 'rl_agents.agents.tree_search.graph_based_stochastic.StochasticGraphBasedPlannerAgent'>",
        "gamma": gamma,
        "upper_bound":
        {
            "type": "kullback-leibler",
            "threshold": "0*np.log(time)",
            "transition_threshold": "0*np.log(time)"
        },
        "max_next_states_count": 1,
        "accuracy": 1e-2
    },
    "MDP-GapE": {
        "__class__": "<class 'rl_agents.agents.tree_search.mdp_gape.MDPGapEAgent'>",
        "gamma": gamma,
        "accuracy": 0,
        "confidence": 1,
        "upper_bound":
        {
            "type": "kullback-leibler",
            "time": "global",
            "threshold": "0*np.log(time)",
            "transition_threshold": "0*np.log(time)"
        },
        "max_next_states_count": 1,
        "continuation_type": "uniform",
        "step_strategy": "reset",
    },
    "BRUE": {
        "__class__": "<class 'rl_agents.agents.tree_search.brue.BRUEAgent'>",
        "gamma": gamma,
        "step_strategy": "reset",
    },
    "UCT": {
        "__class__": "<class 'rl_agents.agents.tree_search.mcts.MCTSAgent'>",
        "gamma": gamma,
        "closed_loop": True
    },
}


def evaluate(env, agent_name, budget, seed=None):
    print("Evaluating", agent_name, "with budget", budget)
    agent_config = agents[agent_name]
    agent_config["budget"] = budget
    agent = agent_factory(env, agent_config)
    if seed is not None:
        env.seed(seed)
        agent.seed(seed)
    obs = env.reset()
    agent.act(obs)
    return agent


def evaluate_agents(env, agents, budget, seed=None):
    for agent_name in agents.keys():
        agent = evaluate(env, agent_name, budget, seed=seed)
        yield agent, agent_name

def score(observations):
    start, goal = np.array([0, 0]), np.array([10, 10])
    score = np.mean([np.linalg.norm(start - obs)
                     - np.linalg.norm(goal - obs)
                     + np.linalg.norm(goal - start)
                     for obs in observations])
    return score


def compare_agents(env, agents, budget, seed=None, show_tree=False, show_trajs=False, show_states=False,
                   show_scores=True):
    trajectories = {}
    state_occupations = {}
    state_updates = {}
    state_limits = 20
    data = {}
    for agent, agent_name in evaluate_agents(env, agents, budget, seed):
        trajectories[agent_name] = agent.planner.root.get_trajectories(env)
        # Aggregate visits
        visits = agent.planner.get_visits()
        updates = agent.planner.get_updates()

        if isinstance(env, GridEnv):
            state_occupations[agent_name] = np.zeros((2 * state_limits + 1, 2 * state_limits + 1))
            state_updates[agent_name] = np.zeros((2 * state_limits + 1, 2 * state_limits + 1))
            for i, x in enumerate(np.arange(-state_limits, state_limits)):
                for j, y in enumerate(np.arange(-state_limits, state_limits)):
                    state_occupations[agent_name][i, j] = visits[str(np.array([x, y]))]
                    state_updates[agent_name][i, j] = updates[str(np.array([x, y]))]
            data[agent_name] = {
                "agent": rename(agent_name),
                "kind": "deterministic" if agent_name[-1] == "D" else "stochastic",
                "ours": agent_name[:4] == "GBOP",
                "score": score(agent.planner.observations),
                "observations": len(agent.planner.observations)
            }

        if show_tree:
            TreePlot(agent.planner, max_depth=12).plot(out / "tree_{}.pdf".format(agent_name),
                                                        title=rename(agent_name))
            plt.show()

    if show_states:
        v_max = max([st.max() for st in state_occupations.values()] + [0])
        if v_max > 0:
            for agent_name, occupations in state_occupations.items():
                show_state_map("occupations", agent_name, occupations, state_limits, v_max)
        # v_max = max([st.max() for st in state_updates.values()] + [0])
        # if v_max > 0:
        #     for agent_name, updates in state_updates.items():
        #         show_state_map("updates", agent_name, updates, state_limits, v_max)

    if show_trajs:
        axes = None
        palette = itertools.cycle(sns.color_palette())
        for agent_name, agent_trajectories in trajectories.items():
            axes = show_trajectories(agent_name, agent_trajectories, axes=axes, color=next(palette))
        plt.show()
        plt.savefig(out / "trajectories.png")

    if show_scores:
        data = pd.DataFrame(list(data.values()))
        data = data.sort_values(["score"])
        ax = data.plot.bar(x='agent', y='score', rot=0, figsize=[8, 4.8], legend=False)
        ax.xaxis.set_label_text("")
        ax.get_figure().savefig(out / "score.pdf")


def show_state_map(title, agent_name, values, state_limits, v_max=None):
    fig, ax = plt.subplots()
    img = ax.imshow(values.T,
                    extent=(-state_limits, state_limits, -state_limits, state_limits),
                    norm=colors.SymLogNorm(linthresh=1, linscale=1, vmax=v_max),
                    cmap=plt.cm.coolwarm)
    fig.colorbar(img, ax=ax)
    plt.grid(False)
    plt.title(rename(agent_name))
    plt.savefig(out / "{}_{}.pdf".format(title, agent_name))
    plt.show()


def show_trajectories(agent_name, trajectories, axes=None, color=None):
    if not axes:
        fig, axes = plt.subplots()
        for trajectory in trajectories:
            x, y = zip(*trajectory.observation)
            plt.plot(x, y, linestyle='dotted', linewidth=0.5, label=rename(agent_name), color=color)
    return axes


def rename(value, latex=True):
    latex_names = {
        "simple_regret": "simple regret $r_n$",
        "total_reward": "total reward $R$",
        "mean_return": "mean return $E[R]$",
        "1/epsilon": r"${1}/{\epsilon}$",
        "mdp-gape-conf": r"\texttt{MDP-GapE}",
        "MDP-GapE": r"\texttt{MDP-GapE}",
        "KL-OLOP": r"\texttt{KL-OLOP}",
        "OPD": r"\texttt{OPD}",
        "BRUE": r"\texttt{BRUE}",
        "GBOP": r"\texttt{GBOP}",
        "GBOP-D": r"\texttt{GBOP-D}",
        "GBOP-T": r"\texttt{GBOP-D}",
        "UCT": r"\texttt{UCT}",
        "budget": r"budget $n$",
    }
    return latex_names.get(value, value) if latex else value


if __name__ == "__main__":
    configure("configs/logging.json", gym_level=gym.logger.INFO)
    selected_env = load_environment(envs["gridenv"])
    selected_agents = [
         "OPD",
         "GBOP-D",
         # "GBOP-T",
         "KL-OLOP",
         "MDP-GapE",
         "GBOP",
         "BRUE",
         "UCT",
    ]
    selected_agents = {k: v for k, v in agents.items() if k in selected_agents}
    budget = 4 * (4 ** 6 - 1) / (4 - 1)
    compare_agents(selected_env, selected_agents, budget=budget,
                   show_tree=False, show_states=True, show_trajs=False, seed=2)
