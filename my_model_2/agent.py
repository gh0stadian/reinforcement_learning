import random

import numpy as np
import torch

from dataset import Experience


class Agent:
    def __init__(self, env, replay_buffer, state_transform=None, action_transform=None):
        self.env = env
        self.replay_buffer = replay_buffer
        self.state_transform = state_transform
        self.action_transform = action_transform
        self.state = None

        self.state_vector = []
        self. reward_vector = []

        self.reset()

    def reset(self):
        state = self.env.reset()
        self.state_vector = [state, state, state, state]

        # WAIT FOR ZOOM
        for i in range(50):
            state, reward, done, _ = self.env.step([0, 0, 0])
            self.state_vector.pop(0)
            state = self.state_transform(state)
            self.state_vector.append(state)

        self.state = np.stack(self.state_vector, axis=0)

    def get_action(self, net, epsilon, device):
        if np.random.random() < epsilon:
            action = np.random.rand(5)

        else:
            state = np.expand_dims(self.state, axis=0)
            state = torch.tensor(state)
            if device not in ["cpu"]:
                state = state.type('torch.FloatTensor').cuda(device)

            action = net(state).squeeze().cpu().numpy()
        return action

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        action = self.get_action(net, epsilon, device)

        total_reward = 0
        done = False
        for i in range(4):
            self.state_vector[i], reward, is_done, _ = self.env.step(self.action_transform(action))
            total_reward += reward
            done = True if is_done else False
            self.state_vector[i] = self.state_transform(self.state_vector[i])

        new_state = np.stack(self.state_vector, axis=0)

        exp = Experience(self.state, action, total_reward, done, new_state)

        self.replay_buffer.append(exp)
        self.state = new_state

        if done:
            self.reset()

        return total_reward, done
