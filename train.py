import numpy as np
from ple import PLE
from ple.games.waterworld import WaterWorld
from tensorflow import keras
import time
import random
from collections import deque

game = WaterWorld(
    height=320, width=320, num_creeps=5
)  # create our game

fps = 120  # fps we want to run at
frame_skip = 2
num_steps = 1
force_fps = True
display_screen = True

# make a PLE instance.
p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps, force_fps=force_fps, display_screen=display_screen)
p.init()

learning_rate = 0.01
momentum = 0.9

def agent(state_shape, action_shape):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=state_shape))
    model.add(keras.layers.Dense(100, activation='relu', kernel_regularizer='l2'))
    model.add(keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'))
    model.add(keras.layers.Dense(action_shape, activation='linear'))
    model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.1),
                  metrics=['accuracy'])
    return model


discount_factor = 0.4
min_replay_size = 500
batch_size = 50


def train(replay_memory, model, target_model):
    if len(replay_memory) < min_replay_size:
        return

    batch = random.sample(replay_memory, batch_size)

    current_state_list = np.array([e[0] for e in batch])
    current_q_list = model.predict(current_state_list)

    future_state_list = np.array([e[3] for e in batch])
    future_q_list = target_model.predict(future_state_list)

    for index, (state, action, reward, future_state, done) in enumerate(batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_q_list[index])
        else:
            max_future_q = reward

        current_q_list[index][action] = max_future_q

    model.fit(current_state_list, current_q_list, batch_size=batch_size, verbose=0, shuffle=True)


def preprocess_state(state):
    processed_state = []
    processed_state.append(state['player_x'])
    processed_state.append(state['player_y'])
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


train_episodes = 5000
frames_per_episode = 1001

epsilon = 1  # we've designed an epsilon policy such that the first 200 episodes, epsilon goes from 1 to 0, then, every 400 episodes, we start a mini-exploration phase of 50 episodes
max_epsilon = 1
min_epsilon = 0
decay = 0.005

state_shape = len(preprocess_state(p.getGameState()))
action_shape = len(p.getActionSet())
action_set = p.getActionSet()

model = agent(state_shape, action_shape)
target_model = agent(state_shape, action_shape)
replay_memory = deque(maxlen=500_000)

steps = 0

for episode in range(train_episodes):
    start_time = time.time()
    score = 0
    p.reset_game()
    state = preprocess_state(p.getGameState())

    for frame in range(frames_per_episode):
        rand_nb = np.random.rand()
        if rand_nb <= epsilon:  # explore
            action = p.getActionSet()[np.random.randint(0, action_shape)]
        else:  # exploit action with max q
            processed_input = state.reshape([1, state_shape])
            predicted_qs = model.predict(processed_input)
            action = action_set[np.argmax(predicted_qs[0])]

        reward = p.act(action)
        next_state = preprocess_state(p.getGameState())
        done = p.game_over()

        # hugging the walls is discouraged
        if state[0] == 0 or state[0] == 290:
            reward -= 0.1
        if state[1] == 0 or state[1] == 290:
            reward -= 0.1

        score += reward

        replay_memory.append([state, action_set.index(action), reward, next_state, done])

        # train every 4 frames
        if frame % 4 == 0 or done:
            train(replay_memory, model, target_model)

        state = next_state
        steps += 1

        # every 500 steps we update the target model
        if steps >= 500:
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

    if epsilon > min_epsilon:
        epsilon = epsilon - decay

    if episode > 0 and episode % 350 == 0:
        print("Started mini-exploration")
        epsilon = max_epsilon
        decay = 0.02

    model.save("model.h5")
    print("Model saved to disk")
    target_model.save("target_model.h5")
    print("Target model saved to disk")
