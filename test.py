import numpy as np
from ple import PLE
from ple.games.waterworld import WaterWorld
from tensorflow import keras
from pprint import pprint
import random
from collections import deque


game = WaterWorld(
    height=320, width=320, num_creeps=5
)  # create our game

fps = 30  # fps we want to run at
frame_skip = 2
num_steps = 1
force_fps = False
display_screen = True


# make a PLE instance.
p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps, force_fps=force_fps, display_screen=display_screen)
p.init()


def agent(model_file):
    return keras.models.load_model(model_file)


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


state_shape = len(preprocess_state(p.getGameState()))
action_shape = len(p.getActionSet())

model = agent('model.h5')

test_episodes = 10

state = preprocess_state(p.getGameState())

for episode in range(test_episodes):
    print('Episode {}'.format(episode))
    score = 0
    p.reset_game()
    while not p.game_over():
        # print('\nplayer x', state[0])
        # print('player y', state[1])
        processed_input = state.reshape([1, state_shape])
        predicted_qs = model.predict(processed_input).flatten()
        action = p.getActionSet()[np.argmax(predicted_qs)]
        reward = p.act(action)
        next_state = preprocess_state(p.getGameState())
        done = p.game_over()
        state = next_state
        score += reward
        if done:
            print("Episode {} score: {}".format(episode, score))
            break

