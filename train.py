import numpy as np
from ple import PLE
from ple.games.waterworld import WaterWorld
import tensorflow as tf
from tensorflow import keras
import time
import random
from collections import deque
from keras.layers import Dense
from keras.regularizers import l2
from keras.losses import Huber

game = WaterWorld(
    height=320, width=320, num_creeps=5
)  

fps = 30  
frame_skip = 1
num_steps = 1
force_fps = True
display_screen = True

p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps, force_fps=force_fps, display_screen=display_screen)
p.init()

learning_rate = 0.001
kernel_reg = l2()
loss_function = Huber()
optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)


def agent(state_shape, action_shape):
    """
        Method that creates a neural network which has as input a processed game state and as output a list of qs for each possible action.
        The neural network has three hidden layers and one output layer:
        - layer 1
            - 100 units
            - receives as input state_shape values
            - activation - ReLU
            - regularizer - L2
        - layer 2
            - 100 units
            - activation - ReLU
            - regularizer - L2
        - layer 3
            - 50 units
            - activation - ReLU
            - regularizer - L2
        - output layer
            - action_shape units
            - activation linear 
        Optimizer - Adam
        Loss function - Huber, custom (see train method)
        :param state_shape: shape of a state, basically the shape of the input
        :param action_shape: shape the action set, basically the shape of the output
        :return: model described above
    """
    model = keras.Sequential()
    model.add(Dense(100, activation='relu', input_shape=(state_shape,), kernel_regularizer=kernel_reg))
    model.add(Dense(100, activation='relu', kernel_regularizer=kernel_reg))
    model.add(Dense(50, activation='relu', kernel_regularizer=kernel_reg))
    model.add(Dense(action_shape, activation='linear'))
    return model


# def train(replay_memory, model, target_model):
#     if len(replay_memory) < batch_size:
#         return
#
#     batch = random.sample(replay_memory, batch_size)
#
#     current_state_list = np.array([e[0] for e in batch])
#     current_q_list = model.predict(current_state_list)
#
#     future_state_list = np.array([e[3] for e in batch])
#     future_q_list = target_model.predict(future_state_list)
#
#     for index, (state, action, reward, future_state, done) in enumerate(batch):
#         if not done:
#             max_future_q = reward + discount_factor * np.max(future_q_list[index])
#         else:
#             max_future_q = reward
#
#         current_q_list[index][action] = max_future_q
#
#     model.fit(current_state_list, current_q_list, batch_size=batch_size, verbose=0)

# def train(replay_memory, model, target_model):
#     if len(replay_memory) < batch_size:
#         return
#
#     # now
#     #  - we only predict if future state is not final
#     #  - we predict both current_q and future_q using the target model
#     #  - we fit on model for each element in batch, not fot the whole batch
#
#     batch = random.sample(replay_memory, batch_size)
#
#     for sample in batch:
#         state, action, reward, future_state, done = sample
#         current_q_list = model.predict(state.reshape([1, state_shape]))
#         if not done:
#             future_q_max = max(target_model.predict(future_state)[0])
#             current_q_list[0][action] = reward + discount_factor * future_q_max
#         else:
#             current_q_list[0][action] = reward
#
#
#         model.fit(state, current_q_list, epochs=1, verbose=0)

def preprocess_state(state):
    """
        Method that processes a given state (which has a lot of information), by keeping only some essential information of a given game state:
        - player position x
        - player position y
        - player velocity x
        - player velocity y
        - closest green x
        - closest green y
        - closest red x
        - closest red y
        :param state: state that is to be processed
        :return: numpy array, representing a processed state, containing the values mentioned above
    """
    processed_state = []
    processed_state.append(state['player_x'])
    processed_state.append(state['player_y'])
    processed_state.append(state['player_velocity_x'])
    processed_state.append(state['player_velocity_y'])
    if len(state['creep_dist']['GOOD']) != 0:
        index_closest_good = np.argmin(np.array(state['creep_dist']['GOOD']))
        processed_state.extend(state['creep_pos']['GOOD'][index_closest_good])
    else:
        processed_state.extend([0, 0])

    if len(state['creep_dist']['BAD']) != 0:
        index_closest_bad = np.argmin(np.array(state['creep_dist']['BAD']))
        processed_state.extend(state['creep_pos']['BAD'][index_closest_bad])
    else:
        processed_state.extend([0, 0])
    return np.array(processed_state)


discount_factor = 0.8
# min_replay_size = 500
batch_size = 100
state_shape = len(preprocess_state(p.getGameState()))
action_shape = len(p.getActionSet())


def train(replay_memory, model, model_target):
    """
        Having the replay memory, train method samples a batch from it and trains the main model.
        :param replay_memory: deque storing past experiences of the agent
        :param model: main model, used for predicting Q-values for current state
        :param model_target: target model, used for predicting maximum future Q-values, basically targets for training the main model
    """
    
    # We only train if we have enough elements in the replay memory to sample a batch
    if len(replay_memory) < batch_size:
        return

    batch = random.sample(replay_memory, batch_size)

    state_sample = np.array([e[0] for e in batch])
    action_sample = [e[1] for e in batch]
    rewards_sample = [e[2] for e in batch]
    state_next_sample = np.array([e[3] for e in batch])
    done_sample = tf.convert_to_tensor([float(e[4]) for e in batch])

    # Predict the rewards from future states using our target model for stability
    future_rewards = model_target.predict(np.array(state_next_sample))

    # Q-value for future rewards = reward + discount factor * maximum future reward => target for training the main model
    future_q_values = rewards_sample + discount_factor * tf.reduce_max(
        future_rewards, axis=1
    )

    # Set the future Q-values to -1 for final states
    future_q_values = future_q_values * (1 - done_sample) - done_sample

    # Create a mask so we only calculate loss on the updated Q-values, corresponding to an action
    masks = tf.one_hot(action_sample, action_shape)

    with tf.GradientTape() as tape:
        # Predict Q-values for all the states in the batch using the main model
        q_values = model(state_sample)

        # Apply the masks to the Q-values to get the Q-value for each action in the batch
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

        # Calculate loss between new target (future Q-value) and predicted Q-value for each action in sample => custom loss function
        loss = loss_function(future_q_values, q_action)

    # Backpropagation on the main model
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


train_episodes = 5000
# An episode ends when the player gets all the green creeps, or when we reach 1000 frames
frames_per_episode = 1001

epsilon = 1  # We've designed an epsilon policy such that in the first 500 episodes, epsilon goes from max_epsilon to min_epsilon, then, every 100 episodes, we start a mini-exploration phase of 25 episodes
max_epsilon = 1
min_epsilon = 0.1   # This way, our agent will always explore with a probability of 10%. Also, it helps with the training speed, because the agent won't get stuck
decay = 0.002  # Epsilon goes from max_epsilon to min_epsilon in 500 episodes

action_set = p.getActionSet()

# Create our models, one for prediction, that is going to be trained, and one for targets, that is going to be updated every 1000 steps with the weights from the main model
model = agent(state_shape, action_shape)
target_model = agent(state_shape, action_shape)
target_model.set_weights(model.get_weights())

# Replay memory for storing state, reward, action, future_state, done
replay_memory = deque(maxlen=100000)

steps = 0

for episode in range(train_episodes):
    start_time = time.time()
    score = 0
    p.reset_game()
    # Get initial state
    state = preprocess_state(p.getGameState())
    for frame in range(frames_per_episode):
        rand_nb = np.random.rand()
        if rand_nb <= epsilon:  # Explore
            action = p.getActionSet()[np.random.randint(0, action_shape)]
        else:  # Exploit action with maximum predicted Q, using our main model
            processed_input = state.reshape([1, state_shape])
            predicted_qs = model.predict(processed_input)
            action = action_set[np.argmax(predicted_qs[0])]

        # Receive reward for selected action
        reward = p.act(action)
        next_state = preprocess_state(p.getGameState())
        # Check if state is final
        done = p.game_over()

        # Hugging the walls is discouraged
        if state[0] == 0 or state[0] == 290:
            reward -= 0.1
        if state[1] == 0 or state[1] == 290:
            reward -= 0.1

        # Compute a score for an episode
        score += reward

        # Store in the replay memory
        replay_memory.append([state, action_set.index(action), reward, next_state, done])

        # Train every 4 frames
        if frame % 4 == 0 or done:
            train(replay_memory, model, target_model)

        state = next_state
        steps += 1

        # Every 1000 steps we update the target model
        if steps >= 1000:
            print('Copying weights from main model to target model...')
            target_model.set_weights(model.get_weights())
            steps = 0

        if done:
            break

    if done:
        print("\nEpisode {} DONE\n\t- score: {}\n\t- duration: {}\n\t- epsilon: {}".format(episode, score,
                                                                                           time.time() - start_time,
                                                                                           epsilon))
    else:
        print("\nEpisode {} \n\t- score: {}\n\t- duration: {}\n\t- epsilon: {}".format(episode, score,
                                                                                       time.time() - start_time,
                                                                                       epsilon))

    # Update epsilon after each episode
    if epsilon > min_epsilon:
        epsilon = epsilon - decay
    else:
        epsilon = min_epsilon

    # Start a mini-exploration phase
    if episode > 500 and episode % 100 == 0:
        print("Started mini-exploration")
        epsilon = max_epsilon
        decay = 0.02  # so that epsilon goes from max_epsilon to min_epsilon over the span of 25 episodes

    model.save("model.h5")
    print("Model saved to disk")
    target_model.save("target_model.h5")
    print("Target model saved to disk")
