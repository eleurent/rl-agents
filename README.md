# rl-agents

A collection of Reinforcement Learning agents

[![Build Status](https://travis-ci.org/eleurent/rl-agents.svg?branch=master)](https://travis-ci.org/eleurent/rl-agents/)

* [Installation](#installation)
* [Usage](#usage)
* [Agents](#agents)
  * [Value Iteration](#value-iteration)
  * [Robust Value Iteration](#robust-value-iteration)
  * [DQN](#dqn)
  * [Monte-Carlo Tree Search](#monte-carlo-tree-search)
  * [Robust Monte-Carlo Tree Search](#robust-monte-carlo-tree-search)

# Installation

`pip install --user git+https://github.com/eleurent/rl-agents`


# Usage

Most experiments can be run from `scripts/experiments.py`

```
Usage:
  experiments evaluate <environment> <agent> (--train|--test)
                                             [--episodes <count>]
                                             [--seed <str>]
                                             [--analyze]
  experiments benchmark <benchmark> (--train|--test)
                                    [--processes <count>]
                                    [--episodes <count>]
                                    [--seed <str>]
  experiments -h | --help

Options:
  -h --help            Show this screen.
  --analyze            Automatically analyze the experiment results.
  --episodes <count>   Number of episodes [default: 5].
  --processes <count>  Number of running processes [default: 4].
  --seed <str>         Seed the environments and agents.
  --train              Train the agent.
  --test               Test the agent.
```

The `evaluate` command allows to evaluate a given agent on a given environment. For instance,

```bash
# Train a DQN agent on the CartPole-v0 environment
$ python3 experiments.py evaluate envs/cartpole.json agents/dqn.json --train --episodes=200
```

The environments are described by their [gym](https://github.com/openai/gym) registration `id`
```JSON
{
    "id":"CartPole-v0"
}
```

And the agents by their class, and configuration dictionary.

```JSON
{
    "__class__": "<class 'rl_agents.agents.dqn.pytorch.DQNAgent'>",
    "model": {
        "type": "DuelingNetwork",
        "layers": [512, 512]
    },
    "gamma": 0.99,
    "n_steps": 1,
    "batch_size": 32,
    "memory_capacity": 50000,
    "target_update": 1,
    "exploration": {
        "method": "EpsilonGreedy",
        "tau": 50000,
        "temperature": 1.0,
        "final_temperature": 0.1
    }
}
```

If keys are missing from these configurations, default values will be used instead.

Finally, a batch of experiments can be scheduled in a _benchmark_.
All experiments are then executed in parallel on several processes.

```bash
# Run a benchmark of several agents interacting with environments
$ python3 experiments.py benchmark cartpole_benchmark.json --test --processes=4
```

A benchmark configuration files contains a list of environment configurations and a list of agent configurations.

```JSON
{
    "environments": ["envs/cartpole.json"],
    "agents":["agents/dqn.json", "agents/mcts.json"]
}
```



# Agents

The following agents are currently implemented:

## [Value Iteration](rl_agents/agents/dynamics_programming/value_iteration.py)

Perform a Value Iteration to compute the state-action value, and acts greedily with respect to it.

Only compatible with [finite-mdp](https://github.com/eleurent/finite-mdp) environments, or environments that handle an `env.to_finite_mdp()` conversion method.

### [Robust Value Iteration](rl_agents/agents/dynamics_programming/robust_value_iteration.py)

In this variant, a list of possible [finite-mdp] models is provided in the agent configuration, and the corresponding robust state-action value is computed so as to maximize the worst-case total reward.

## [DQN](rl_agents/agents/dqn)

A neural-network model is used to estimate the state-action value function and produce a greedy optimal policy.

Implemented variants:
* Double DQN
* Dueling architecture
* N-step targets

References:
* [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mnih V. et al, 2013
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), van Hasselt H. et al, 2015.
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581), Wang Z. et al, 2015.

## [Monte-Carlo Tree Search](rl_agents/agents/tree_search/mcts.py)

A world transition model is leveraged for trajectory search. A search tree is expanded by efficient random sampling so as to focus the search around the most promising moves.

References:
* [Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search](https://hal.inria.fr/inria-00116992/document), Coulom R., 2006.
* [Bandit based Monte-Carlo Planning](http://ggp.stanford.edu/readings/uct.pdf), Kocsis L., Szepesvári C., 2006.

### [Robust Monte Carlo Tree Search](rl_agents/agents/tree_search/robust_mcts.py)

In this variant, a list of environment modifiers (called preprocessors) is provided in the agent configuration to generate several possible environment, and the corresponding robust state-action value is approximately computed by tree-search so as to maximize the worst-case total reward.



