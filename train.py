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


def agent(state_shape, action_shape):
    learning_rate = 0.003
    init = keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=state_shape))
    model.add(keras.layers.Dense(100, activation='relu', kernel_regularizer='l2', kernel_initializer=init))
    model.add(keras.layers.Dense(100, activation='relu', kernel_regularizer='l2', kernel_initializer=init))
    model.add(keras.layers.Dense(10, activation='relu', kernel_regularizer='l2', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.SGD(learning_rate=learning_rate), metrics=['accuracy'])
    return model


learning_rate = 0.003
discount_factor = 0.8
min_replay_size = 500
batch_size = 10


def train(replay_memory, model, target_model):

    if len(replay_memory) < min_replay_size:
        return

    mini_batch = random.sample(replay_memory, batch_size)
    current_state_list = np.array([e[0] for e in mini_batch])
    current_q_list = model.predict(current_state_list)
    future_state_list = np.array([e[3] for e in mini_batch])
    future_q_list = target_model.predict(future_state_list)

    x = []
    y = []
    for index, (state, action, reward, future_state, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_q_list[index])
        else:
            max_future_q = reward

        current_q = current_q_list[index]
        current_q[action] = (1 - learning_rate) * current_q[action] + learning_rate * max_future_q

        x.append(state)
        y.append(current_q)

    model.fit(np.array(x), np.array(y), batch_size=batch_size, verbose=0, shuffle=True)


def preprocess_state(state):
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

    if len(state['creep_dist']['BAD']) !=0:
        index_closest_bad = np.argmin(np.array(state['creep_dist']['BAD']))
        processed_state.extend(state['creep_pos']['BAD'][index_closest_bad])
    else:
        processed_state.extend([0, 0])
    return np.array(processed_state)


train_episodes = 5000
frames_per_episode = 1001
epsilon = 1  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
max_epsilon = 1  # You can't explore more than 100% of the time
min_epsilon = 0.01  # At a minimum, we'll always explore 1% of the time
decay = 0.01

state_shape = len(preprocess_state(p.getGameState()))
action_shape = len(p.getActionSet())

model = agent(state_shape, action_shape)
target_model = agent(state_shape, action_shape)
replay_memory = deque(maxlen=500_000)
steps_to_update_target_model = 0

for episode in range(train_episodes):
    start_time = time.time()
    score = 0
    p.reset_game()
    state = preprocess_state(p.getGameState())
    for frame in range(frames_per_episode):
        steps_to_update_target_model += 1

        rand_nb = np.random.rand()
        if rand_nb <= epsilon:  # explore
            action = p.getActionSet()[np.random.randint(0, action_shape)]
        else:  # exploit action with max q
            processed_input = state.reshape([1, state_shape])
            predicted_qs = model.predict(processed_input).flatten()
            action = p.getActionSet()[np.argmax(predicted_qs)]

        reward = p.act(action)
        next_state = preprocess_state(p.getGameState())
        done = p.game_over()

        #hugging the walls is discouraged
        if state[0] == 0 or state[0] == 290:
            reward -= 0.1
        if state[1] == 0 or state[1] == 290:
            reward -= 0.1

        score += reward

        replay_memory.append([state, p.getActionSet().index(action), reward, next_state, done])

        #train every 4 frames
        if frame % 4 == 0 or done:
            train(replay_memory, model, target_model)

        state = next_state

        #after each episode that ended by consuming all good creeps, we update our target model if 400 steps have passed
        if done:
            if steps_to_update_target_model >= 400:
                print('Copying weights from main model to target model...')
                target_model.set_weights(model.get_weights())
                steps_to_update_target_model = 0
            break

    if done:
        print("\nEpisode {} DONE\n\t- score: {}\n\t- duration: {}".format(episode, score, time.time()-start_time))
    else:
        print("\nEpisode {} \n\t- score: {}\n\t- duration: {}".format(episode, score, time.time() - start_time))

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

    model.save("model.h5")
    print("Model saved to disk")
    target_model.save("target_model.h5")
    print("Target model saved to disk")
