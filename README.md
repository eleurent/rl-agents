# rl-agents

A collection of Reinforcement Learning agents

![build](https://github.com/eleurent/rl-agents/workflows/build/badge.svg)

* [Installation](#installation)
* [Usage](#usage)
* [Monitoring](#monitoring)
* [Agents](#agents)
  * Planning
    * [Value Iteration](#vi-value-iteration)
    * [Cross-Entropy Method](#cem-cross-entropy-method)
    * Monte-Carlo Tree Search
      * [Upper Confidence Trees](#uct-upper-confidence-bounds-applied-to-trees)
      * [Deterministic Optimistic Planning](#opd-optimistic-planning-for-deterministic-systems)
      * [Open Loop Optimistic Planning](#olop-open-loop-optimistic-planning)
      * [Trailblazer](#trailblazer)
      * [PlaTγPOOS](#plaTγpoos)
  * Safe planning
    * [Robust Value Iteration](#rvi-robust-value-iteration)
    * [Discrete Robust Optimistic Planning](#drop-discrete-robust-optimistic-planning)
    * [Interval-based Robust Planning](#irp-interval-based-robust-planning)
  * Value-based
    * [Deep Q-Network](#dqn-deep-q-network)
    * [Fitted-Q](#ftq-fitted-q)
  * Safe value-based
    * [Budgeted Fitted-Q](#bftq-budgeted-fitted-q)
* [Citing](#citing) 

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
$ python3 experiments.py evaluate configs/CartPoleEnv/env.json configs/CartPoleEnv/DQNAgent.json --train --episodes=200
```

Every agent interacts with the environment following a standard interface:
```python
action = agent.act(state)
next_state, reward, done, info = env.step(action)
agent.record(state, action, reward, next_state, done, info)
```

The environments are described by their [gym](https://github.com/openai/gym) `id`, and module for registration.
```JSON
{
    "id": "CartPole-v0",
    "import_module": "gym"
}
```

And the agents by their class, and configuration dictionary.

```JSON
{
    "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
    "model": {
        "type": "MultiLayerPerceptron",
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

If keys are missing from these configurations, values in `agent.default_config()` will be used instead.

Finally, a batch of experiments can be scheduled in a _benchmark_.
All experiments are then executed in parallel on several processes.

```bash
# Run a benchmark of several agents interacting with environments
$ python3 experiments.py benchmark cartpole_benchmark.json --test --processes=4
```

A benchmark configuration file contains a list of environment configurations and a list of agent configurations.

```JSON
{
    "environments": ["envs/cartpole.json"],
    "agents": ["agents/dqn.json", "agents/mcts.json"]
}
```

# Monitoring

There are several tools available to monitor the agent performances:
* *Run metadata*: for the sake of reproducibility, the environment and agent configurations used for the run are merged and saved to a `metadata.*.json` file.
* [*Gym Monitor*](https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py): the main statistics (episode rewards, lengths, seeds) of each run are logged to an `episode_batch.*.stats.json` file. They can be automatically visualised by running `scripts/analyze.py`
* [*Logging*](https://docs.python.org/3/howto/logging.html): agents can send messages through the standard python logging library. By default, all messages with log level _INFO_ are saved to a `logging.*.log` file. Add the option `scripts/experiments.py --verbose` to save with log level _DEBUG_.
* [*Tensorboard*](https://github.com/lanpa/tensorboardX): by default, a tensoboard writer records information about useful scalars, images and model graphs to the run directory. It can be visualized by running:
```tensorboard --logdir <path-to-runs-dir>```

# Agents

The following agents are currently implemented:

## Planning

### [`VI` Value Iteration](rl_agents/agents/dynamic_programming/value_iteration.py)

Perform a Value Iteration to compute the state-action value, and acts greedily with respect to it.

Only compatible with [finite-mdp](https://github.com/eleurent/finite-mdp) environments, or environments that handle an `env.to_finite_mdp()` conversion method.

Reference: [Dynamic Programming](https://press.princeton.edu/titles/9234.html), Bellman R., Princeton University Press (1957).

### [`CEM` Cross-Entropy Method](rl_agents/agents/cross_entropy_method/cem.py)

A sampling-based planning algorithm, in which sequences of actions are drawn from a prior gaussian distribution. This distribution is iteratively bootstraped by minimizing its cross-entropy to a target distribution approximated by the top-k candidates.

Only compatible with continuous action spaces. The environment is used as an oracle dynamics and reward model. 

Reference: [A Tutorial on the Cross-Entropy Method](web.mit.edu/6.454/www/www_fall_2003/gew/CEtutorial.pdf), De Boer P-T., Kroese D.P, Mannor S. and Rubinstein R.Y. (2005).

### `MCTS` Monte-Carlo Tree Search

A world transition model is leveraged for trajectory search. A look-ahead tree is expanded so as to explore the trajectory space and quickly focus around the most promising moves.

References:
* [Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search](https://hal.inria.fr/inria-00116992/document), Coulom R., 2006.

#### [`UCT` Upper Confidence bounds applied to Trees](rl_agents/agents/tree_search/mcts.py)
The tree is traversed by iteratively applying an optimistic selection rule at each depth, and the value at leaves is estimated by sampling.
Empirical evidence shows that this popular algorithms performs well in many applications, but it has been proved theoretically to achieve a much worse performance (doubly-exponential) than uniform planning in some problems.

References:
* [Bandit based Monte-Carlo Planning](http://ggp.stanford.edu/readings/uct.pdf), Kocsis L., Szepesvári C. (2006).
* [Bandit Algorithms for Tree Search](https://hal.inria.fr/inria-00136198v2), Coquelin P-A., Munos R. (2007).

#### [`OPD` Optimistic Planning for Deterministic systems](rl_agents/agents/tree_search/deterministic.py)
This algorithm is tailored for systems with deterministic dynamics and rewards.
It exploits the reward structure to achieve a polynomial rate on regret, and behaves efficiently in numerical experiments with dense rewards.

Reference: [Optimistic Planning for Deterministic Systems](https://hal.inria.fr/hal-00830182), Hren J., Munos R. (2008).

#### [`OLOP` Open Loop Optimistic Planning](rl_agents/agents/tree_search/olop.py)

References: 
* [Open Loop Optimistic Planning](http://sbubeck.com/COLT10_BM.pdf), Bubeck S., Munos R. (2010).
* [Practical Open-Loop Optimistic Planning](https://arxiv.org/abs/1904.04700), Leurent E., Maillard O.-A. (2019).

#### [Trailblazer](rl_agents/agents/tree_search/trailblazer.py)

Reference: [Blazing the trails before beating the path: Sample-efficient Monte-Carlo planning](http://researchers.lille.inria.fr/~valko/hp/serve.php?what=publications/grill2016blazing.pdf), Grill J. B., Valko M., Munos R. (2017).

#### [PlaTγPOOS](rl_agents/agents/tree_search/platypoos.py)

Reference: [Scale-free adaptive planning for deterministic dynamics & discounted rewards](http://researchers.lille.inria.fr/~valko/hp/publications/bartlett2019scale-free.pdf), Bartlett P., Gabillon V., Healey J., Valko M. (2019).


## Safe planning

### [`RVI` Robust Value Iteration](rl_agents/agents/dynamic_programming/robust_value_iteration.py)

A list of possible [finite-mdp](https://github.com/eleurent/finite-mdp) models is provided in the agent configuration. The MDP ambiguity set is constrained to be rectangular: different models can be selected at every transition.The corresponding robust state-action value is computed so as to maximize the worst-case total reward.

References:
* [Robust Control of Markov Decision Processes with Uncertain Transition Matrices](https://people.eecs.berkeley.edu/~elghaoui/pdffiles/rmdp_erl.pdf), Nilim A., El Ghaoui L. (2005).
* [Robust Dynamic Programming](http://www.corc.ieor.columbia.edu/reports/techreports/tr-2002-07.pdf), Iyengar G. (2005).
* [Robust Markov Decision Processes](http://www.optimization-online.org/DB_FILE/2010/05/2610.pdf), Wiesemann W. et al. (2012).

### [`DROP` Discrete Robust Optimistic Planning](rl_agents/agents/tree_search/robust.py)

The MDP ambiguity set is assumed to be finite, and is constructed from a list of modifiers to the true environment.
The corresponding robust value is approximately computed by [Deterministic Optimistic Planning](#deterministic-optimistic-planning) so as to maximize the worst-case total reward.

References:
* [Approximate Robust Control of Uncertain Dynamical Systems](https://arxiv.org/abs/1903.00220), Leurent E. et al. (2018).

### [`IRP` Interval-based Robust Planning](rl_agents/agents/tree_search/robust.py)

We assume that the MDP is a parametrized dynamical system, whose parameter is uncertain and lies in a continuous ambiguity set. We use interval prediction to compute the set of states that can be reached at any time _t_, given that uncertainty, and leverage it to evaluate and improve a robust policy.

If the system is Linear Parameter-Varying (LPV) with polytopic uncertainty, an fast and stable interval predictor can be designed. Otherwise, sampling-based approaches can be used instead, with an increased computational load.

References:
* [Approximate Robust Control of Uncertain Dynamical Systems](https://arxiv.org/abs/1903.00220), Leurent E. et al. (2018).
* [Interval Prediction for Continuous-Time Systems with Parametric Uncertainties](https://arxiv.org/abs/1904.04727), Leurent E. et al (2019).

## Value-based

### [`DQN` Deep Q-Network](rl_agents/agents/deep_q_network)

A neural-network model is used to estimate the state-action value function and produce a greedy optimal policy.

Implemented variants:
* Double DQN
* Dueling architecture
* N-step targets

References:
* [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mnih V. et al (2013).
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), van Hasselt H. et al. (2015).
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581), Wang Z. et al. (2015).

### [`FTQ` Fitted-Q](rl_agents/agents/fitted_q)

A Q-function model is trained by performing each step of Value Iteration as a supervised learning procedure applied to a batch
of transitions covering most of the state-action space.

Reference: [Tree-Based Batch Mode Reinforcement Learning](http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf), Ernst D. et al (2005).

## Safe Value-based

### [`BFTQ` Budgeted Fitted-Q](rl_agents/agents/budgeted_ftq)

An adaptation of **`FTQ`** in the budgeted setting: we maximise the expected reward _r_ of a policy _π_ under the constraint that an expected cost _c_ remains under a given budget _β_.
The policy _π(a | s, _β_)_ is conditioned on this cost budget _β_, which can be changed online.

To that end, the Q-function model is trained to predict both the expected reward _Qr_ and the expected cost _Qc_ of the optimal constrained policy _π_. 

This agent can only be used with environments that provide a cost signal in their `info` field:
```
>>> obs, reward, done, info = env.step(action)
>>> info
{'cost': 1.0}
``` 

Reference: [Budgeted Reinforcement Learning in Continuous State Space](https://arxiv.org/abs/1903.01004), Carrara N., Leurent E., Laroche R., Urvoy T., Maillard O-A., Pietquin O. (2019).

# Citing

If you use this project in your work, please consider citing it with:
```
@misc{rl-agents,
  author = {Leurent, Edouard},
  title = {rl-agents: Implementations of Reinforcement Learning algorithms},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/rl-agents}},
}
```
