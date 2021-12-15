import decimal
import re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.colors import ListedColormap

f = open('Training Attempt #8\' - 1073 ep - PROMISING RESULTS/log.txt', 'r')
data = f.read()

episode_regex = re.compile('Episode (\d+)')
episodes = np.array(episode_regex.findall(data)).astype(int)
# print(episodes)

score_regex = re.compile('score: (-*\d+.*\d*)')
scores = np.array(score_regex.findall(data)).astype(float)
# print(scores)

duration_regex = re.compile('duration: (\d+.*\d*)')
durations = np.array(duration_regex.findall(data)).astype(float)
# print (durations)

epsilon_regex = re.compile('epsilon: (\d+.*\d*)')
epsilon = np.array(epsilon_regex.findall(data)).astype(float)

episodes_done_regex = re.compile('Episode (\d+) DONE')
episodes_done = np.array(episodes_done_regex.findall(data)).astype(int)

done = np.zeros(len(episodes))
for episode in episodes:
    if episode in episodes_done:
        done[episode] = 1

# figure(figsize=(40, 5), dpi=80)
# color_map = ListedColormap(['red', 'darkblue'])
# plt.scatter(episodes, np.zeros(len(episodes)), c=done, cmap=color_map)
# plt.ylim(-0.1, 0.1)


# plt.plot(episodes, epsilon)

figure(figsize=(30, 15), dpi=80)
plt.plot(episodes, scores)
plt.plot(episodes, durations)

plt.show()
