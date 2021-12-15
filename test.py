import numpy as np
from ple import PLE
from ple.games.waterworld import WaterWorld
from tensorflow import keras

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


def agent(model_file):
    """
    Loads the model from a given path.
    :param model_file: location on the dist where the model was saved
    :return: the model
    """
    return keras.models.load_model(model_file)

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


state_shape = len(preprocess_state(p.getGameState()))
action_shape = len(p.getActionSet())
model = agent('model.h5')
test_episodes = 10

for episode in range(test_episodes):
    print('Episode {}'.format(episode))
    score = 0
    p.reset_game()
    state = preprocess_state(p.getGameState())
    while not p.game_over():
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

