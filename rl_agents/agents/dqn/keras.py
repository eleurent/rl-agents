import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam

from rl_agents.agents.dqn.abstract import AbstractDQNAgent
from rl_agents.agents.dqn.keras_invariant import PermutationInvariant


class DQNAgent(AbstractDQNAgent):
    def __init__(self, env, config):
        super(DQNAgent, self).__init__(env, config)
        self.model = None
        self.build_perm_net()

    def record(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.optimize_model()

    def build_neural_net(self):
        """
            Build a neural network model for value function approximation
        """
        self.model = Sequential()

        # Input layer
        self.model.add(Dense(self.config["model"]["all_layers"][1], kernel_initializer='lecun_uniform', input_shape=(self.config["model"]["all_layers"][0],)))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.2))

        for layer in self.config["model"]["all_layers"][2:-1]:
            self.model.add(Dense(layer, kernel_initializer='lecun_uniform'))
            self.model.add(Activation('tanh'))
            self.model.add(Dropout(0.2))

        # Output layer
        self.model.add(Dense(self.config["model"]["all_layers"][-1], kernel_initializer='lecun_uniform'))
        self.model.add(Activation('linear'))

        optim = Adam(lr=5e-4)
        self.model.compile(loss='mse', optimizer=optim)

    def build_perm_net(self):
        inp_shape = (5, 5)
        layer_sizes = self.config["model"]["all_layers"][1:]
        tuple_dim = 2
        self.model = PermutationInvariant(input_shape=inp_shape,
                                        layer_sizes=layer_sizes,
                                        tuple_dim=tuple_dim,
                                        reduce_fun="mean")

        optim = Adam(lr=5e-4)
        self.model.compile(loss='mse', optimizer=optim)

    def optimize_model(self):
        try:
            minibatch = self.memory.sample(self.config['batch_size'])
        except ValueError:
            return

        X_train = []
        y_train = []
        for memory in minibatch:
            state, action, reward, next_state, terminal = memory

            # Get the current prediction of action values
            state = np.reshape(state, (5, -1))
            qvalues = self.model.predict(np.array([state]), batch_size=1)
            y = np.zeros((1, self.config['num_actions']))
            y[:] = qvalues[:]

            # Get our predicted best next move
            next_state= np.reshape(next_state, (5, -1))
            nextQ = self.model.predict(np.array([next_state]), batch_size=1)
            # Update the value for the action we took.
            if not terminal:
                value = reward + self.config['gamma'] * np.max(nextQ)
            else:
                value = reward
            y[0][action] = value

            X_train.append(state.reshape((5,-1)))
            y_train.append(y.reshape(self.config['num_actions'],))

        self.model.fit(np.array(X_train), np.array(y_train), batch_size=self.config['batch_size'], epochs=1, verbose=0)

    def get_batch_state_values(self, states):
        action_values = self.get_batch_state_action_values(states)
        return np.max(action_values, 1), np.argmax(action_values, 1)

    def get_batch_state_action_values(self, states):
        states = np.reshape(states, (1, 5, -1))
        return self.model.predict(np.array(states), batch_size=1)

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()
