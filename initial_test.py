import numpy as np
from ple import PLE
from ple.games.waterworld import WaterWorld
import time
from pprint import pprint
class NaiveAgent():
    """
            This is our naive agent. It picks actions at random!
    """

    def __init__(self, actions):
        self.actions = actions

    def pickAction(self):
        return self.actions[np.random.randint(0, len(self.actions))]

###################################
game = WaterWorld(
    height=640, width=480
)  # create our game

fps = 30  # fps we want to run at
frame_skip = 2
num_steps = 1
force_fps = False  # slower speed
display_screen = True

reward = 0.0
max_noops = 20
nb_frames = 15000

# make a PLE instance.
p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
        force_fps=force_fps, display_screen=display_screen)

# our Naive agent!
agent = NaiveAgent(p.getActionSet())

# init agent and game.
p.init()

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

print (p.getActionSet())