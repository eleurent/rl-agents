import torch
from torch.distributions import Normal

from rl_agents.agents.common.abstract import AbstractAgent
from rl_agents.agents.common.factory import safe_deepcopy_env


class CEMAgent(AbstractAgent):
    """
        Cross-Entropy Method planner.
        The environment is copied and used as an oracle model to sample trajectories.
    """
    def __init__(self, env, config):
        super(CEMAgent, self).__init__(config)
        self.env = env
        self.action_size = env.action_space.shape[0]

    @classmethod
    def default_config(cls):
        return dict(gamma=1.0,
                    horizon=10,
                    iterations=10,
                    candidates=100,
                    top_candidates=10)

    def plan(self, observation):
        action_distribution = Normal(
            torch.zeros(self.config["horizon"], self.action_size),
            torch.ones(self.config["horizon"], self.action_size))
        for i in range(self.config["iterations"]):
            # Evaluate J action sequences from the current belief (in batch)
            actions = action_distribution.sample([self.config["candidates"]])  # Sample actions
            candidates = [safe_deepcopy_env(self.env) for _ in range(self.config["candidates"])]
            returns = torch.zeros(self.config["candidates"])
            # Sample next states
            for t in range(self.config["horizon"]):
                for c, candidate in enumerate(candidates):
                    _, reward, _, _ = candidate.step(actions[c, t])
                    returns[c] += self.config["gamma"]**t * reward

            # Re-fit belief to the K best action sequences
            _, topk = returns.topk(self.config["top_candidates"], largest=True, sorted=False)  # K ← argsort({R(j)}
            best_actions = actions[topk]
            # Update belief with new means and standard deviations
            action_distribution = Normal(best_actions.mean(dim=0), best_actions.std(dim=0, unbiased=False))
        # Return first action mean µ_t
        return action_distribution.mean.tolist()

    def record(self, state, action, reward, next_state, done, info):
        pass

    def act(self, state):
        return self.plan(state)[0]

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, filename):
        return False

    def load(self, filename):
        return False


