from __future__ import division, print_function
import numpy as np
import copy


class MaxNode(object):
    def __init__(self, state, gamma, delta, alpha, eta, depth=0):
        self.state = state
        self.gamma = gamma
        self.delta = delta
        self.alpha = alpha
        self.eta = eta
        self.K = state.action_space.n
        self.depth = depth

        self.children = {}
        for action in range(state.action_space.n):
            self.children[action] = AvgNode(state, action, self.gamma, self.delta, self.alpha, self.eta, self.K, self.depth + 1)

    def run(self, m, epsilon):
        candidates = self.children.values()
        L = 1
        U = 1/(1-self.gamma)
        mu = []

        while len(candidates) > 1 and U >= (1 - self.eta)*epsilon:
            sqr = (np.log(self.K*L/(self.delta*epsilon)) +
                   self.gamma / (self.eta - self.gamma) + self.alpha + 1) / L
            U = 2/(1-self.gamma)*np.sqrt(sqr)
            if self.depth == 0:
                print("U={} / {}".format(U, (1 - self.eta)*epsilon))
            mu = [(b, b.run(L, U*self.eta/(1-self.eta))) for b in candidates]
            mu_sup = max(mu, key=lambda c: c[1])[1]
            candidates = [c[0] for c in mu if c[1] + 2*U/(1-self.eta) >= mu_sup - 2*U/(1-self.eta)]
            L += 1

        if len(candidates) > 1:
            return max(mu, key=lambda c: c[1])[1]
        else:
            return candidates[0].run(m, self.eta*epsilon)

    def __eq__(self, other):
        # TODO: generic comparison for list.index()
        return self.state.mdp.state == other.state.mdp.state


class AvgNode(object):
    oracle_calls = 1

    def __init__(self, state, action, gamma, delta, alpha, eta, K, depth):
        self.state = state
        self.action = action
        self.gamma = gamma
        self.delta = delta
        self.alpha = alpha
        self.eta = eta
        self.K = K
        self.depth = depth

        self.sampled_nodes = []
        self.r = 0

    def run(self, m, epsilon):
        if epsilon >= 1/(1-self.gamma):
            return 0
        if len(self.sampled_nodes) >= m:
            active_nodes = self.sampled_nodes[:m]
        else:
            while len(self.sampled_nodes) < m:
                new_state = copy.deepcopy(self.state)
                _, new_reward, _, _ = new_state.step(self.action)
                self.sampled_nodes.append(MaxNode(new_state, self.gamma, self.delta, self.alpha, self.eta, self.depth + 1))
                AvgNode.oracle_calls += 1
                self.r += new_reward
            active_nodes = self.sampled_nodes
        # At this point, |active_nodes| == m

        uniques = []
        counts = []
        for s in active_nodes:
            try:
                i = uniques.index(s)
                counts[i] += 1
            except ValueError:
                uniques.append(s)
                counts.append(1)

        mu = 0
        for i in range(len(uniques)):
            nu = uniques[i].run(counts[i], epsilon/self.gamma)
            mu += counts[i]/m*nu
        return self.r/len(self.sampled_nodes) + self.gamma*mu


class TrailBlazer(object):
    def __init__(self, state, gamma, delta, epsilon):
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.eta = np.power(self.gamma, 1/max(2, np.log(1/self.epsilon)))
        self.K = state.action_space.n
        self.alpha = 2*np.log(self.epsilon*(1-self.gamma))**2 * \
            np.log(np.log(self.K)/(1-self.eta)) / np.log(self.eta/self.gamma)
        self.alpha = 0
        self.m = (np.log(1/self.delta) + self.alpha) / ((1 - self.gamma) ** 2 * self.epsilon ** 2)
        print('gamma {}'.format(gamma))
        print('delta {}'.format(delta))
        print('epsilon {}'.format(epsilon))
        print('self.eta {}'.format(self.eta))
        print('self.K {}'.format(self.K))
        print('self.alpha {}'.format(self.alpha))
        print('self.m {}'.format(self.m))

        self.root = MaxNode(state, gamma, delta, self.alpha, self.eta)

    def run(self):
        return self.root.run(self.m, self.epsilon/2)


def test():
    import finite_mdp
    import gym
    env = gym.make('finite-mdp-v0')
    env.configure({
        "mode": "deterministic",
        "transition": [[1, 2],
                       [1, 1],
                       [2, 2],
                       [3, 3]],
        "reward": [[0.5, 1],
                   [0, 0],
                   [0, 0],
                   [0, 0]]
    })
    env.reset()

    tb = TrailBlazer(env, gamma=0.5, delta=0.1, epsilon=4.0)
    print(tb.run())


if __name__ == '__main__':
    test()
