import gym
import keyboard
import matplotlib.pyplot as plt
import torchvision.transforms as trans
import numpy as np

env = gym.make("LunarLander-v2")
observation = env.reset()

done = False
while not done:
    action = 0
    if keyboard.is_pressed("d"):
        action = 1
    if keyboard.is_pressed("w"):
        action = 2
    if keyboard.is_pressed("a"):
        action = 3
    #
    observation, reward, done, _ = env.step(action)
    # get_screen(observation)
    # print("REWARD IS", reward)
    # step = env.action_space.sample()
    # env.step(action)
    print(reward)
    env.render(mode='human')

exit(0)