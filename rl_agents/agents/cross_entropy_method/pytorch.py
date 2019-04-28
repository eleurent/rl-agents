import torch
from torch.distributions import Normal

from rl_agents.agents.cross_entropy_method.cem import CEMAgent


class PytorchCEMAgent(CEMAgent):
    """
    CEM planner with Recurrent state-space models (RSSM) for transition and rewards, as in PlaNet.
    Original implementation by Kai Arulkumaran from https://github.com/Kaixhin/PlaNet/blob/master/planner.py
    Allows batch forward of many candidates (e.g. 1000)
    """
    def __init__(self, env, config, transition_model, reward_model):
        super(CEMAgent, self).__init__(config)
        self.env = env
        self.action_size = env.action_space.shape[0]
        self.transition_model = transition_model
        self.reward_model = reward_model

    def plan(self, belief, state):
        belief, state = belief.expand(self.config["candidates"], -1), state.expand(self.config["candidates"], -1)
        # Initialize factorized belief over action sequences q(a_t:t+H) ← N(0, I)
        action_distribution = Normal(torch.zeros(self.config["horizon"], self.action_size, device=belief.device),
                                     torch.ones(self.config["horizon"], self.action_size, device=belief.device))
        for i in range(self.config["iterations"]):
            # Evaluate J action sequences from the current belief (in batch)
            beliefs, states = [belief], [state]
            actions = action_distribution.sample([self.config["candidates"]])  # Sample actions
            # Sample next states
            for t in range(self.config["horizon"]):
                next_belief, next_state, _, _ = self.transition_model(states[-1], actions[:, t], beliefs[-1])
                beliefs.append(next_belief)
                states.append(next_state)
            # Calculate expected returns (batched over time x batch)
            beliefs = torch.stack(beliefs[1:], dim=0).view(self.config["horizon"] * self.config["candidates"], -1)
            states = torch.stack(states[1:], dim=0).view(self.config["horizon"] * self.config["candidates"], -1)
            returns = self.reward_model(beliefs, states).view(self.config["horizon"], self.config["candidates"]).sum(dim=0)
            # Re-fit belief to the K best action sequences
            _, topk = returns.topk(self.config["top_candidates"], largest=True, sorted=False)  # K ← argsort({R(j)}
            best_actions = actions[topk]
            # Update belief with new means and standard deviations
            action_distribution = Normal(best_actions.mean(dim=0), best_actions.std(dim=0, unbiased=False))
        # Return first action mean µ_t
        return action_distribution.mean[0].to_list()
