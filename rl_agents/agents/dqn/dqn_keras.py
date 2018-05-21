import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam

from rl_agents.agents.dqn.abstract import DqnAgent
from rl_agents.agents.utils import ReplayMemory
from rl_agents.agents.exploration.exploration import ExplorationPolicy


class DqnKerasAgent(DqnAgent):
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
        self.optimize_model()

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

        self.model.fit(np.array(X_train), np.array(y_train), batch_size=self.config['batch_size'], epochs=1, verbose=0)

    def get_batch_state_value(self, states):
        action_values = self.get_batch_state_action_values(states)
        return np.max(action_values), np.argmax(action_values)

    def get_batch_state_action_values(self, states):
        return self.model.predict(np.array(states), batch_size=1)

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