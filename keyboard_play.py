import gym
import keyboard
import matplotlib.pyplot as plt
import torchvision.transforms as trans
import numpy as np

env = gym.make("CarRacing-v1")
observation = env.reset()

done = False
for i in range(50):
    observation, reward, done, _ = env.step([0, 0, 0])
while not done:
    action = [0, 0, 0.1]
    if keyboard.is_pressed("w"):
        action[1] = 0.7
    if keyboard.is_pressed("s"):
        action[2] = 1
    if keyboard.is_pressed("a"):
        action = [-1, 0, 0]
    if keyboard.is_pressed("d"):
        action = [1, 0, 0]

    observation, cumulative_reward, done, _ = env.step(action)
    for i in range(3):
        action = [element * 0.7 for element in action]
        observation, reward, done, _ = env.step(action)
        cumulative_reward += reward

    print("REWARD IS", cumulative_reward)
    env.render()

exit(0)
