import numpy as np
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam


class DQN:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.config["numStates"] = env.observation_space.shape[0]
        self.config["numActions"] = env.action_space.n
        self.config["neuralNet"] = [self.config["numStates"]] + self.config["neuralNet"] + [self.config["numActions"]]
        self.model = None
        self.replay = []
        self.epsilon = self.config['epsilon'][0]
        self.build_neural_net()

    def train(self):
        """
            Train the model to take actions in an environment
            and maximize its rewards
        """
        max_score = -float("inf")

        for episode in range(self.config['episodes']):
            state = self.env.reset()
            score = 0
            done = False
            while not done:
                self.env.render()
                # Take action.
                action = self.pick_action(state)
                # Step environment.
                prev_state = state
                state, reward, done, info = self.env.step(action)
                score += reward
                # Experience replay.
                self.store_experience((prev_state, action, reward, state, done))
                minibatch = self.sample_replay_memory()
                # Fit the model on this batch.
                if minibatch:
                    X_train, y_train = self.process_minibatch(minibatch)
                    self.model.fit(X_train, y_train, batch_size=self.config['batchSize'], nb_epoch=1, verbose=0)

            # End of episode
            if score > max_score:
                max_score = score
                self.model.save('last_best.h5')
            print("Episode %d score: %f (max %f)\tepsilon %f" %
                  (episode, score, max_score, self.epsilon))

    def pick_action(self, state):
        """
            Pick an action with an epsilon-greedy policy
        """
        if self.epsilon > self.config['epsilon'][1]:
            self.epsilon -= (self.config['epsilon'][0]-self.config['epsilon'][1])/self.config['tau']
        if random.random() < self.epsilon:
            action = np.random.randint(self.config['numActions'])
        else: # Predict best action with model
            action = np.argmax(self.model.predict(np.array([state]), batch_size=1))
        return action

    def store_experience(self, sample):
        """
            Store experience in replay memory
        """
        self.replay.append(sample)
        if len(self.replay) > self.config['memory']:
            self.replay.pop(0)

    def sample_replay_memory(self):
        """
            Randomly sample our experience replay memory
        """
        if len(self.replay) > self.config['batchSize']:
            return random.sample(self.replay, self.config['batchSize'])
        else:
            return None

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
            y = np.zeros((1, self.config['numActions']))
            y[:] = qvalues[:]

            # Get our predicted best next move
            nextQ = self.model.predict(np.array([next_state]), batch_size=1)
            # Update the value for the action we took.
            if not terminal:
                value = reward + self.config['gamma'] * np.max(nextQ)
            else:
                value = reward
            y[0][action] = value

            X_train.append(state.reshape(self.config['numStates'],))
            y_train.append(y.reshape(self.config['numActions'],))

        return np.array(X_train), np.array(y_train)

    def build_neural_net(self):
        """
            Build a neural network model for value function approximation
        """
        self.model = Sequential()
        layers = self.config["neuralNet"]

        # First layer
        self.model.add(Dense(layers[1], init='lecun_uniform', input_shape=(layers[0],)))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.2))

        for layer in layers[2:-1]:
            self.model.add(Dense(layer, init='lecun_uniform'))
            self.model.add(Activation('tanh'))
            self.model.add(Dropout(0.2))

        # Output layer
        self.model.add(Dense(layers[-1], init='lecun_uniform'))
        self.model.add(Activation('linear'))

        optim = Adam(lr=5e-4)
        self.model.compile(loss='mse', optimizer=optim)