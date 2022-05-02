import torch
import wandb
import math
from tqdm import tqdm
from collections import OrderedDict
from typing import List, Tuple

from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from agent_fixed import Agent
from config import train_config, model_config, env_config
from dataset_fixed import ReplayBuffer
from config import *

wandb.init(project="RL-car", entity="jbdb", config={'train': train_config, 'model': model_config, 'env': env_config})


class Trainer:
    def __init__(self, model, target_model, env, device, action_transform=None, state_transform=None, config=None):
        self.eps_start = config['eps_start']
        self.eps_end = config['eps_end']
        self.eps_decay = config['eps_decay']

        self.lr = config['lr']
        self.gamma = config['gamma']
        self.sync_rate = config['sync_rate']
        self.reward_decreasing_limit = config['reward_decreasing_limit']
        self.replay_size = config['replay_size']
        self.batch_size = config['batch_size']

        self.device = device
        self.net = model.to(device)

        wandb.watch(self.net, log_freq=10)

        self.target_net = target_model.to(device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.state_transform = state_transform
        self.action_transform = action_transform

        self.buffer = ReplayBuffer(self.replay_size)
        self.agent = Agent(env,
                           self.buffer,
                           state_transform=state_transform,
                           action_transform=action_transform,
                           step_length=4)

        self.reward_decreasing_counter = 0
        self.total_reward = 0
        self.episode_reward = 0
        self.episodes = 0

        self.populate(config['warm_start_steps'])

    def populate(self, steps):
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def calc_loss(self, batch, device):
        states, actions, rewards, dones, next_states = batch
        ones_vector = torch.ones(len(dones)).to(device)
        gamma = torch.FloatTensor([self.gamma] * len(dones)).to(device)

        state_action_values = self.net(states)
        state_action_values = state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states)
            next_state_values = next_state_values.max(1)[0]

        dones = (ones_vector - dones)
        expected_state_action_values = (next_state_values * gamma * dones) + rewards

        return self.loss(state_action_values, expected_state_action_values)

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def train(self):
        max_episodes = 10000
        constantly_decreasing_reward = 0
        self.net.train()

        for current_episode in tqdm(range(0, max_episodes)):
            score = 0
            length = 0
            done = False

            while not done:

                epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * current_episode / self.eps_decay)
                reward, done = self.agent.play_step(self.net, epsilon, self.device)
                batch = self.buffer.sample(self.batch_size, self.device)
                loss = self.calc_loss(batch, self.device)

                score += reward
                length += 1

                self.optimizer.zero_grad()
                loss.backward()
                for param in self.net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

                if reward < 0:
                    constantly_decreasing_reward += 1
                else:
                    constantly_decreasing_reward = 0

                if constantly_decreasing_reward > 30:
                    done = True
                    self.agent.reset()

            # swap params
            if current_episode % self.sync_rate == 0:
                self.target_net.load_state_dict(self.net.state_dict())

            if current_episode % 20 == 0:
                print("Recording video......")
                video = self.agent.record_play(self.net, self.device)
                wandb.log({"video": wandb.Video(video.swapaxes(3, 2).swapaxes(2, 1), fps=24)}, step=current_episode)

            wandb.log({"episode/length": length,
                       "episode/reward": score,
                       "episode/count": current_episode
                       },
                      step=current_episode)
