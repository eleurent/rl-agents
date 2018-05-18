import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam

from rl_agents.agents.abstract import AbstractAgent
from rl_agents.agents.utils import ReplayMemory
from rl_agents.agents.exploration.exploration import ExplorationPolicy


class DqnKerasAgent(AbstractAgent):
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.config["num_states"] = env.observation_space.shape[0]
        self.config["num_actions"] = env.action_space.n
        self.config["layers"] = [self.config["num_states"]] + self.config["layers"] + [self.config["num_actions"]]
        self.memory = ReplayMemory(self.config)
        self.exploration_policy = ExplorationPolicy(config)
        self.model = None
        self.build_neural_net()

    def record(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        try:
            minibatch = self.memory.sample(self.config['batch_size'])
        except ValueError:
            minibatch = None
        # Fit the model on this batch.
        if minibatch:
            X_train, y_train = self.process_minibatch(minibatch)
            self.model.fit(X_train, y_train, batch_size=self.config['batch_size'], epochs=1, verbose=0)

    def act(self, state):
        """
            Pick an action with an epsilon-greedy policy
        """
        optimal_action = np.argmax(self.model.predict(np.array([state]), batch_size=1))
        return self.exploration_policy.epsilon_greedy(optimal_action, self.env.action_space)

    def process_minibatch(self, minibatch):
        """
            Use Bellman optimal equation to update the Q values
            of transitions stored in the minibatch
        """
        X_train = []
        y_train = []
        for memory in minibatch:
            state, action, reward, next_state, terminal = memory

            # Get the current prediction of action values
            qvalues = self.model.predict(np.array([state]), batch_size=1)
            y = np.zeros((1, self.config['num_actions']))
            y[:] = qvalues[:]

            # Get our predicted best next move
            nextQ = self.model.predict(np.array([next_state]), batch_size=1)
            # Update the value for the action we took.
            if not terminal:
                value = reward + self.config['gamma'] * np.max(nextQ)
            else:
                value = reward
            y[0][action] = value

            X_train.append(state.reshape(self.config['num_states'],))
            y_train.append(y.reshape(self.config['num_actions'],))

        return np.array(X_train), np.array(y_train)

    def build_neural_net(self):
        """
            Build a neural network model for value function approximation
        """
        self.model = Sequential()
        layers = self.config["layers"]

        # Input layer
        self.model.add(Dense(layers[1], kernel_initializer='lecun_uniform', input_shape=(layers[0],)))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.2))

        for layer in layers[2:-1]:
            self.model.add(Dense(layer, kernel_initializer='lecun_uniform'))
            self.model.add(Activation('tanh'))
            self.model.add(Dropout(0.2))

        # Output layer
        self.model.add(Dense(layers[-1], kernel_initializer='lecun_uniform'))
        self.model.add(Activation('linear'))

        optim = Adam(lr=5e-4)
        self.model.compile(loss='mse', optimizer=optim)