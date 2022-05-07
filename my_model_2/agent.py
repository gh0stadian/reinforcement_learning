import random

import numpy as np
import torch

from dataset import Experience
from env_wrappers import FireResetEnv, MaxAndSkipEnv, ClipRewardEnv


class Agent:
    def __init__(self, env, replay_buffer, state_transform=None, wrappers_config=None, state_size=4):
        self.env = self.create_env(env, wrappers_config)
        self.replay_buffer = replay_buffer
        self.state_transform = state_transform
        self.state = None
        self.state_size = state_size
        self.state_vector = []
        self.reset()

    def create_env(self, env, wrappers_config):
        if wrappers_config['fire_reset']:
            env = FireResetEnv(env)
        if wrappers_config['max_n_skip']:
            env = MaxAndSkipEnv(env)
        if wrappers_config['clip_reward']:
            env = ClipRewardEnv(env)
        return env

    def reset(self):
        state = self.env.reset()
        self.state_vector = [self.state_transform(state)] * self.state_size
        self.state = np.stack(self.state_vector, axis=0)

    def get_action(self, net, epsilon, device):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()

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

        state, reward, done, _ = self.env.step(action)

        self.state_vector.pop(0)
        state = self.state_transform(state)
        self.state_vector.append(state)

        new_state = np.stack(self.state_vector, axis=0)
        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)
        self.state = new_state

        if done:
            self.reset()

        return reward, done

    def record_play(self, net, device="cpu"):
        recording = []
        done = False

        self.reset()
        while not done:
            action = self.get_action(net, 0.0, device)
            state, reward, done, _ = self.env.step(action)
            recording.append(state)

            self.state_vector.pop(0)
            state = self.state_transform(state)
            self.state_vector.append(state)

            self.state = np.stack(self.state_vector, axis=0)

        self.reset()
        recording = np.stack(recording, axis=0)
        return recording
