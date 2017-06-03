import gym
import numpy as np
# from nn import neural_net, LossHistory

NUM_INPUT = 4
NUM_ACTIONS = 2
GAMMA = 0.9  # Forgetting.

Kx = [0.5, 1]
Ka = [5, 10]
env = gym.make('CartPole-v0')
env.reset()
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        ux = Kx[0]*(0 - observation[0]) - Kx[1]*observation[1]
        ua = Ka[0]*(ux - observation[2]) - Ka[1]*observation[3]
        action = 1 if ua < 0 else 0
        observation, reward, done, info = env.step(action)
        # if done:
            # print("Episode finished after {} timesteps".format(t+1))
            # break


# # import random
# # import os.path
# # import timeit
# # from random import randint

# def train_net(model, params):
#     filename = params_to_filename(params)

#     observe = 1000  # Number of frames to observe before training.
#     epsilon = 1
#     train_frames = 25000  # Number of frames to play.
#     batchSize = params['batchSize']
#     buffer = params['buffer']
#     replay = []  # stores tuples of (S, A, R, S').

#     # Create a new sim instance.
# 	env = gym.make('CartPole-v0')
# 	env.reset()

#     start_time = timeit.default_timer()
#     # Run the frames.
#     for t in range(train_frames):

#         # Choose an action.
#         if random.random() < epsilon or t < observe:
#             action = np.random.randint(NUM_ACTIONS) # random
#         else:
#             # Get Q values for each action.
#             qval = model.predict(state, batch_size=1)
#             action = (np.argmax(qval))  # best

#         # Take action, observe new state and get our treat.
#         state, reward, done, info = env.step(action)
#         print state, action, reward

#         # Experience replay storage.
#         replay.append((prev_state, action, reward, state))
#         prev_state = state

#         # If we're done observing, start training.
#         if t > observe:

#             # If we've stored enough in our buffer, pop the oldest.
#             if len(replay) > buffer:
#                 replay.pop(0)

#             # Randomly sample our experience replay memory
#             minibatch = random.sample(replay, batchSize)

#             # Get training values.
#             X_train, y_train = process_minibatch(minibatch, model)

#             # Train the model on this batch.
#             history = LossHistory()
#             model.fit(
#                 X_train, y_train, batch_size=batchSize,
#                 nb_epoch=1, verbose=0, callbacks=[history]
#             )
#             # loss_log.append(history.losses)

#         # Decrement epsilon over time.
#         if epsilon > 0.1 and t > observe:
#             epsilon -= (1./train_frames)

# #         # if t % 100 == 0:
# #         #     print t, new_state[0,1], action, reward, epsilon

# #         # We died, so update stuff.
# #         if new_state[0,1] < 0:
# #             # Log the simulation duration at this t.
# #             data_collect.append([t, simulation_score])

# #             # Update max.
#             if simulation_score > max_simulation_score:
#                 max_simulation_score = simulation_score

#             # Time it.
#             tot_time = timeit.default_timer() - start_time
#             fps = simulation_score / tot_time

#             # Output some stuff so we can watch.
#             print("Max: %f at %d\tepsilon %f\t(%f)\t%f fps" %
#                   (max_simulation_score, t, epsilon, simulation_score, fps))

#             model.save('saved-models/last_best.h5')

#             # Reset.
#             simulation_score = 0
#             start_time = timeit.default_timer()
#             drone.reset()

#         # Save the model every 25,000 frames.
#         if t % 25000 == 0:
#             save_model(filename, t)

#     # Log results after we're done all frames.
#     log_results(filename, data_collect, loss_log)
#     save_model(filename, t)

# def save_model(filename, t):
#     model.save('saved-models/' + filename + '-' +
#                    str(t) + '.h5')
#     model.save('saved-models/latest.h5')
#     print("Saving model %s - %d" % (filename, t))

# def log_results(filename, data_collect, loss_log):
#     # Save the results to a file so we can graph it later.
#     with open('results/obstacle-frames/learn_data-' + filename + '.csv', 'w') as data_dump:
#         wr = csv.writer(data_dump)
#         wr.writerows(data_collect)

#     with open('results/obstacle-frames/loss_data-' + filename + '.csv', 'w') as lf:
#         wr = csv.writer(lf)
#         for loss_item in loss_log:
#             wr.writerow(loss_item)


# def process_minibatch(minibatch, model):
#     """This does the heavy lifting, aka, the training. It's super jacked."""
#     X_train = []
#     y_train = []
#     # Loop through our batch and create arrays for X and y
#     # so that we can fit our model at every step.
#     for memory in minibatch:
#         # Get stored values.
#         old_state_m, action_m, reward_m, new_state_m = memory
#         # Get prediction on old state.
#         old_qval = model.predict(old_state_m, batch_size=1)
#         # Get prediction on new state.
#         newQ = model.predict(new_state_m, batch_size=1)
#         # Get our best move. I think?
#         maxQ = np.max(newQ)
#         y = np.zeros((1, NUM_ACTIONS))
#         y[:] = old_qval[:]
#         # Check for terminal state.
#         if reward_m != -500:  # non-terminal state
#             update = (reward_m + (GAMMA * maxQ))
#         else:  # terminal state
#             update = reward_m
#         # Update the value for the action we took.
#         y[0][action_m] = update
#         X_train.append(old_state_m.reshape(NUM_INPUT,))
#         y_train.append(y.reshape(NUM_ACTIONS,))

#     X_train = np.array(X_train)
#     y_train = np.array(y_train)

#     return X_train, y_train


# def params_to_filename(params):
#     return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
#             str(params['batchSize']) + '-' + str(params['buffer'])


# def launch_learn(params):
#     filename = params_to_filename(params)
#     print("Trying %s" % filename)
#     # Make sure we haven't run this one.
#     if not os.path.isfile('results/obstacle-frames/loss_data-' + filename + '.csv'):
#         # Create file so we don't double test when we run multiple
#         # instances of the script at the same time.
#         open('results/obstacle-frames/loss_data-' + filename + '.csv', 'a').close()
#         print("Starting test.")
#         # Train.
#         model = neural_net(NUM_INPUT, params['nn'])
#         train_net(model, params)
#     else:
#         print("Already tested.")


# if __name__ == "__main__":
#     if TUNING:
#         param_list = []
#         nn_params = [[164, 150, 3], [256, 256, 3],
#                      [512, 512], [1000, 1000]]
#         batchSizes = [40, 100, 400]
#         buffers = [10000, 50000]

#         for nn_param in nn_params:
#             for batchSize in batchSizes:
#                 for buffer in buffers:
#                     params = {
#                         "batchSize": batchSize,
#                         "buffer": buffer,
#                         "nn": nn_param
#                     }
#                     param_list.append(params)

#         for param_set in param_list:
#             launch_learn(param_set)

#     else:
#         # nn_param = [164, 150, NUM_ACTIONS]
#         nn_param = [100, 100, NUM_ACTIONS]
#         params = {
#             "batchSize": 40,
#             "buffer": 10000,
#             "nn": nn_param
#         }
#         model = neural_net(NUM_INPUT, nn_param)
#         train_net(model, params)
