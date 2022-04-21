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
        self.reset()

    def reset(self):
        state = self.env.reset()

        # WAIT FOR ZOOM
        for i in range(50):
            state, reward, done, _ = self.env.step([0, 0, 0])

        state = self.state_transform(state)
        self.state_vector = [state, state, state, state]
        self.state = np.stack(self.state_vector, axis=0)

    def get_action(self, net, epsilon, device):
        if np.random.random() < epsilon:
            # action = self.env.action_space.sample()
            action = random.choice([0, 1, 2, 3, 4])

        else:
            state = np.expand_dims(self.state, axis=0)
            state = torch.tensor(state)
            if device not in ["cpu"]:
                state = state.type('torch.FloatTensor').cuda(device)

            action = net(state).squeeze().cpu().numpy().argmax()
        return action

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        action = self.get_action(net, epsilon, device)

        new_state, reward, done, _ = self.env.step(self.action_transform(action))
        new_state = self.state_transform(new_state)

        self.state_vector.pop(0)
        self.state_vector.append(new_state)
        new_state = np.stack(self.state_vector, axis=0)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)
        self.state = new_state

        if done:
            self.reset()

        return reward, done
