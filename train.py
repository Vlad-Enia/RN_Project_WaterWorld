import numpy as np
from ple import PLE
from ple.games.waterworld import WaterWorld
from tensorflow import keras
from pprint import pprint
import random
from collections import deque

game = WaterWorld(
    height=640, width=480
)  # create our game

fps = 30  # fps we want to run at
frame_skip = 2
num_steps = 1
force_fps = False  # slower speed
display_screen = True


# make a PLE instance.
p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps, force_fps=force_fps, display_screen=display_screen)
p.init()


def agent(state_shape, action_shape):
    learning_rate = 0.03
    init = keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=state_shape))
    model.add(keras.layers.Dense(state_shape * 2, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.SGD(learning_rate=learning_rate), metrics=['accuracy'])
    return model


def train(replay_memory, model, target_model):
    learning_rate = 0.03
    discount_factor = 0.6

    MIN_REPLAY_SIZE = 500
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return
    # print("TRAIN")
    batch_size = 10
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

    # processed_state.extend([element * (-1) for element in state['creep_dist']['BAD']])
    # processed_state.extend(state['creep_dist']['GOOD'])
    # for list_ in state['creep_pos']['GOOD']:
    #     processed_state.extend(list_)
    # for list_ in state['creep_pos']['BAD']:
    #     processed_state.extend([element * (-1) for element in list_])
    return np.array(processed_state)


train_episodes = 1000
epsilon = 0.9  # at start everything is random
epsilon_decay = 0.0005

state_shape = len(preprocess_state(p.getGameState()))
action_shape = len(p.getActionSet())

print(p.getActionSet())

model = keras.models.load_model('model.h5')
target_model = keras.models.load_model('model.h5')

replay_memory = deque(maxlen=50_000)

x = []
y = []

steps_to_train_model = 0
state = preprocess_state(p.getGameState())

for episode in range(train_episodes):
    score = 0
    p.reset_game()
    while not p.game_over():
        steps_to_train_model += 1
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
        replay_memory.append([state, p.getActionSet().index(action), reward, next_state, done])

        if steps_to_train_model % 5 == 0 or done:
            train(replay_memory, model, target_model)
        state = next_state
        score += reward

        if done:
            print("Episode {} score: {}".format(episode, score))
            if steps_to_train_model >= 50:
                target_model.set_weights(model.get_weights())
                steps_to_train_model = 0
            break
    if epsilon > 0:
        epsilon -= epsilon_decay
    model.save("model.h5")
    print("Model saved to disk")
    target_model.save("target_model.h5")
    print("Target model saved to disk")
