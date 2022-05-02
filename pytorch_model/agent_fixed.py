import random

import numpy as np
import torch
import wandb

from dataset_fixed import Experience
from config import action_space


class Agent:
    def __init__(self, env, replay_buffer, state_transform=None, action_transform=None, step_length=1, state_size=4):
        self.env = env
        self.replay_buffer = replay_buffer
        self.state_transform = state_transform
        self.action_transform = action_transform
        self.step_length = step_length
        self.state = None
        self.state_size = state_size

        self.state_vector = []

        self.reset()

    def reset(self):
        state = self.env.reset()
        self.state_vector = [state] * self.state_size

        # WAIT FOR ZOOM
        for i in range(50):
            state, reward, done, _ = self.env.step([0, 0, 0])

        for i in range(4):
            state, reward, done, _ = self.env.step([0, 0, 0])
            self.state_vector.pop(0)
            state = self.state_transform(state)
            self.state_vector.append(state)

        self.state = np.stack(self.state_vector, axis=0)

    def get_action(self, net, epsilon, device):
        if np.random.random() < epsilon:
            action = random.randrange(0, len(action_space), 1)

        else:
            state = np.expand_dims(self.state, axis=0)
            state = torch.tensor(state)
            if device not in ["cpu"]:
                state = state.type('torch.FloatTensor').cuda(device)

            action = net(state).argmax().squeeze().cpu().numpy()
        return action

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        action = self.get_action(net, epsilon, device)

        total_reward = 0
        done = False
        for i in range(self.step_length):
            state, reward, is_done, _ = self.env.step(self.action_transform(action))

            self.state_vector.pop(0)
            state = self.state_transform(state)
            self.state_vector.append(state)

            total_reward += reward
            if is_done:
                done = True
                self.reset()
                break

        new_state = np.stack(self.state_vector, axis=0)
        exp = Experience(self.state, action, total_reward, done, new_state)
        self.replay_buffer.append(exp)
        self.state = new_state

        return total_reward, done

    def record_play(self, net, device="cpu"):
        recording = []
        done = False

        self.reset()
        while not done:
            action = self.get_action(net, 0.0, device)
            for i in range(self.step_length):
                state, reward, is_done, _ = self.env.step(self.action_transform(action))

                recording.append(state)

                self.state_vector.pop(0)
                state = self.state_transform(state)
                self.state_vector.append(state)

                if is_done:
                    done = True
                    self.reset()
                    break

        recording = np.stack(recording, axis=0)
        return recording
