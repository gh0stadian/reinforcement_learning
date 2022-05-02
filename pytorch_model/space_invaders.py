import gym
import numpy as np


env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
env.reset()
for _ in range(1000):
    # env.render()
    action = env.action_space.sample() # take a random action
    print(action)
    observation, reward, done, info = env.step(action)
    state = np.dot(observation[..., :], [0.299, 0.587, 0.114])
    state = state[18:, :]
    print(observation)
env.close()
