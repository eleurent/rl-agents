import gym
from gym import wrappers
import numpy as np
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop

EPISODE_DURATION = 200

def train(env, model, config):
    """
        Train the model to take actions in an environment
        and maximize its rewards
    """
    replay = []
    epsilon = config['epsilon']
    max_duration = 0

    for episode in range(config['episodes']):
        state = env.reset()
        duration = 0
        done = False
        while not done:
            duration += 1
            env.render()
            # Take action.
            action, epsilon = pick_action(state, model, epsilon, config)
            # Step environment.
            prev_state = state
            state, reward, done, info = env.step(action)
            terminal = done and not (duration == EPISODE_DURATION)
            # Experience replay.
            store_experience(replay, (prev_state, action, reward, state, terminal), config['memory'])
            minibatch = sample_replay_memory(replay, config['batchSize'])
            # Fit the model on this batch.
            if minibatch:
                X_train, y_train = process_minibatch(minibatch, model, config)
                model.fit(X_train, y_train, batch_size=config['batchSize'], nb_epoch=1, verbose=0)

        # End of episode
        if duration > max_duration:
            max_duration = duration
            model.save('last_best.h5')
        print("Episode %d duration: %d (max: %d)\tepsilon %f" %
              (episode, duration, max_duration, epsilon))

def pick_action(state, model, epsilon, config):
    """
        Pick an action with an epsilon-greedy policy
    """
    epsilon -= epsilon/config['tau']
    if random.random() < epsilon:
        action = np.random.randint(config['numActions'])
    else: # Predict best action with model
        action = np.argmax(model.predict(np.array([state]), batch_size=1))
    return action, epsilon

def store_experience(replay, sample, memory):
    """
        Store experience in replay memory
    """
    replay.append(sample)
    if len(replay) > memory:
        replay.pop(0)

def sample_replay_memory(replay, batch_size):
    """
        Randomly sample our experience replay memory
    """
    if len(replay) > batch_size:
        return random.sample(replay, batch_size)
    else:
        return None

def process_minibatch(minibatch, model, config):
    """
        Use Bellman optimal equation to update the Q values
        of transitions stored in the minibatch
    """
    X_train = []
    y_train = []
    for memory in minibatch:
        state, action, reward, next_state, terminal = memory

       # Get the current prediction of action values
        qvalues = model.predict(np.array([state]), batch_size=1)
        y = np.zeros((1, config['numActions']))
        y[:] = qvalues[:]

        # Get our predicted best next move
        nextQ = model.predict(np.array([next_state]), batch_size=1)
        # Update the value for the action we took.
        if not terminal:
            value = reward + config['gamma'] * np.max(nextQ)
        else:
            value = reward
        y[0][action] = value

        X_train.append(state.reshape(config['numStates'],))
        y_train.append(y.reshape(config['numActions'],))

    return np.array(X_train), np.array(y_train)

def build_neural_net(layers):
    """
        Build a neural network model for value function approximation
    """
    model = Sequential()

    # First layer
    model.add(Dense(layers[1], init='lecun_uniform', input_shape=(layers[0],)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))

    for layer in layers[2:-1]:
        print layer
        model.add(Dense(layer, init='lecun_uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(layers[-1], init='lecun_uniform'))
    model.add(Activation('linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

    return model


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, 'tmp/cartpole-experiment-1', force=True)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    config = {
        "numStates": num_states,
        "numActions": num_actions,
        "neuralNet": [num_states, 100, 100, num_actions],
        "memory": 10000,
        "batchSize": 200,
        "episodes": 150,
        "gamma": 0.9,
        "epsilon": 0.5,
        "tau": 500,
    }

    model = build_neural_net(config['neuralNet'])
    train(env, model, config)
