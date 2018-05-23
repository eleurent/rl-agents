# rl-agents

A collection of Reinforcement Learning agents

[![Build Status](https://travis-ci.org/eleurent/rl-agents.svg?branch=master)](https://travis-ci.org/eleurent/rl-agents/)

## Installation

`pip install --user git+https://github.com/eleurent/rl-agents`

## Usage

```python
import gym
from rl_agents.agents.dqn.dqn_pytorch import DQNPytorchAgent
from rl_agents.trainer.simulation import Simulation

# Make a gym environment
env = gym.make('your-env')
# Create a new agent, here a DQN running on PyTorch
agent = DQNPytorchAgent(env)
# Train the agent and monitor its performances
sim = Simulation(env, agent, num_episodes=200)
sim.train()
```

## Agents

The following agents are currently implemented:

### [Double DQN](rl_agents/agents/dqn)

A neural-network model is used to estimate the state-action value function and produce a greedy optimal policy.

References:
* [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mnih V. et al, 2013
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), van Hasselt H. et al, 2015.

### [Monte Carlo Tree Search](rl_agents/agents/tree_search/mcts.py)

A world transition model is leveraged for trajectory search. A search tree is expanded by efficient random sampling so as to focus the search around the most promising moves.

References:
* [Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search](https://hal.inria.fr/inria-00116992/document), Coulom R., 2006.
* [Bandit based Monte-Carlo Planning](http://ggp.stanford.edu/readings/uct.pdf), Kocsis L., Szepesvári C., 2006.
