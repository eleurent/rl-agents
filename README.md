# rl-agents

A collection of Reinforcement Learning agents

[![Build Status](https://travis-ci.org/eleurent/rl-agents.svg?branch=master)](https://travis-ci.org/eleurent/rl-agents/)

* [Installation](#installation)
* [Usage](#usage)
* [Agents](#agents)
  * Planning
    * [Value Iteration](#value-iteration)
    * [Monte-Carlo Tree Search](#monte-carlo-tree-search)
    * [Deterministic Optimistic Planning](#deterministic-optimistic-planning)
    * [Open Loop Optimistic Planning](#open-loop-optimistic-planning)
    * [Trailblazer](#trailblazer)
  * Robust planning
    * [Robust Value Iteration](#robust-value-iteration)
    * [Discrete Robust Optimistic Planning](#discrete-robust-optimistic-planning)
    * [Interval-based Robust Planning](#interval-based-robust-planning)
  * Value-based
    * [DQN](#dqn)

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

## Planning

### [Value Iteration](rl_agents/agents/dynamics_programming/value_iteration.py)

Perform a Value Iteration to compute the state-action value, and acts greedily with respect to it.

Only compatible with [finite-mdp](https://github.com/eleurent/finite-mdp) environments, or environments that handle an `env.to_finite_mdp()` conversion method.

### [Monte-Carlo Tree Search](rl_agents/agents/tree_search/mcts.py)

Otherwise known as Upper Confidence Trees (UCT). A world transition model is leveraged for trajectory search. A search tree is expanded by efficient random sampling so as to focus the search around the most promising moves.

References:
* [Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search](https://hal.inria.fr/inria-00116992/document), Coulom R., 2006.
* [Bandit based Monte-Carlo Planning](http://ggp.stanford.edu/readings/uct.pdf), Kocsis L., Szepesvári C., 2006.

### [Deterministic Optimistic Planning](rl_agents/agents/tree_search/deterministic.py)

Reference:
* [Optimistic Planning for Deterministic Systems](https://hal.inria.fr/hal-00830182), Hren J., Munos R., 2008.

### [Open Loop Optimistic Planning](rl_agents/agents/tree_search/olop.py)

Reference:
* [Open Loop Optimistic Planning](http://sbubeck.com/COLT10_BM.pdf), Bubeck S., Munos R., 2010.

### [Trailblazer](rl_agents/agents/tree_search/trailblazer.py)

Reference:
* [Blazing the trails before beating the path: Sample-efficient Monte-Carlo planning](http://researchers.lille.inria.fr/~valko/hp/serve.php?what=publications/grill2016blazing.pdf), Grill J. B., Valko M., Munos R., 2017.

## Robust planning

### [Robust Value Iteration](rl_agents/agents/dynamics_programming/robust_value_iteration.py)

A list of possible [finite-mdp](https://github.com/eleurent/finite-mdp) models is provided in the agent configuration. The MDP ambiguity set is constrained to be rectangular: different models can be selected at every transition.The corresponding robust state-action value is computed so as to maximize the worst-case total reward.

References:
* [Robust Control of Markov Decision Processes with Uncertain Transition Matrices](https://people.eecs.berkeley.edu/~elghaoui/pdffiles/rmdp_erl.pdf), Nilim A., El Ghaoui L., 2005.
* [Robust Dynamic Programming](http://www.corc.ieor.columbia.edu/reports/techreports/tr-2002-07.pdf), Iyengar G., 2005.
* [Robust Markov Decision Processes](http://www.optimization-online.org/DB_FILE/2010/05/2610.pdf), Wiesemann W. et al, 2012.

### [Discrete Robust Optimistic Planning](rl_agents/agents/tree_search/robust.py)

The MDP ambiguity set is assumed to be finite, and is constructed from a list of modifiers to the true environment. 
The corresponding robust value is approximately computed by [Deterministic Optimistic Planning](#deterministic-optimistic-planning) so as to maximize the worst-case total reward.

### [Interval-based Robust Planning]((rl_agents/agents/tree_search/robust.py))

TODO

## Value-based

### [DQN](rl_agents/agents/dqn)

A neural-network model is used to estimate the state-action value function and produce a greedy optimal policy.

Implemented variants:
* Double DQN
* Dueling architecture
* N-step targets

References:
* [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mnih V. et al, 2013
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), van Hasselt H. et al, 2015.
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581), Wang Z. et al, 2015.

