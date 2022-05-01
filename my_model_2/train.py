import gym
import torch
import copy
import numpy as np
import datetime
import os
import wandb

from collections import OrderedDict
from typing import List, Tuple

from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from agent import Agent
from dataset import RLDataset, ReplayBuffer
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
            self,
            model: torch.nn.Module,
            target_model: torch.nn.Module,
            env: gym.Env,
            action_transform=None,
            state_transform=None,
            batch_size: int = 16,
            lr: float = 1e-2,
            gamma: float = 0.99,
            sync_rate: int = 10,
            reward_decreasing_limit: int = 10,
            replay_size: int = 1000,
            warm_start_size: int = 1000,
            eps_last_frame: int = 1000,
            eps_start: float = 1.0,
            log_video_epoch: int = 1000,
            eps_end: float = 0.01,
            episode_length: int = 200,
            warm_start_steps: int = 1000,
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'target_model', 'env', 'action_transform', 'state_transform'])
        self.state_transform = state_transform
        self.action_transform = action_transform

        self.env = env
        self.net = model
        self.target_net = target_model

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer, state_transform=state_transform, action_transform=action_transform)
        self.reward_decreasing_counter = 0

        self.total_reward = 0
        self.episode_reward = 0
        self.episodes = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps=1000):
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x):
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch, device):
        states, actions, rewards, dones, next_states = batch
        dones = dones.type('torch.IntTensor').cuda(device)
        states = states.type('torch.FloatTensor').cuda(device)
        next_states = next_states.type('torch.FloatTensor').cuda(device)
        ones_vector = torch.ones(len(dones)).cuda(device)
        actions = actions.argmax(axis=1).type(torch.int64).unsqueeze(-1)
        gamma = torch.FloatTensor([self.hparams.gamma] * len(dones)).cuda(device)

        state_action_values = self.net(states)
        state_action_values = state_action_values.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states)
            next_state_values = next_state_values.max(1)[0]

        dones = (ones_vector-dones)
        expected_state_action_values = (next_state_values * gamma * dones) + rewards

        # penalty = self.calc_not_moving_penalty(states)
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def calc_not_moving_penalty(self, state):
        state_last = state[:, -1]
        state_first = state[:, 0]
        non_zero_diff = (state_last-state_first).count_nonzero(dim=(1, 2))
        non_zero_diff[non_zero_diff < 10] = 4
        non_zero_diff[non_zero_diff >= 10] = 1
        return non_zero_diff


    def training_step(self, batch, nb_batch):
        device = self.get_device(batch)

        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame,
        )

        reward, done = self.agent.play_step(self.net, epsilon, device)
        done = self.check_early_stop(reward)
        self.episode_reward += reward

        loss = self.dqn_mse_loss(batch, device)

        if done:
            self.total_reward = self.episode_reward
            self.episodes += 1
            self.episode_reward = 0
            self.reward_decreasing_counter = 0

        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log("total_reward", torch.tensor(self.total_reward).to(device))
        self.log("reward", torch.tensor(reward).to(device))
        self.log("episode_reward", torch.tensor(self.episode_reward).to(device))
        self.log("train/loss", loss)
        self.log("steps", torch.tensor(self.global_step).to(device))
        self.log("episodes", torch.tensor(self.episodes).to(device))
        return {"loss": loss, "total_reward": self.total_reward}

    def check_early_stop(self, reward):
        done = False
        if reward < 0:
            self.reward_decreasing_counter += 1
            if self.reward_decreasing_counter == self.hparams.reward_decreasing_limit:
                self.reward_decreasing_counter = 0
                done = True
                self.agent.reset()
        else:
            self.reward_decreasing_counter = 0
        return done

    def configure_optimizers(self):
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self):
        return self.__dataloader()

    def get_device(self, batch):
        return batch[0].device.index if self.on_gpu else "cpu"

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.hparams.log_video_epoch == 0:
            state = self.env.reset()

            # WAIT FOR ZOOM
            for i in range(50):
                state, reward, done, _ = self.env.step([0, 0, 0])
            gif = [state]

            state = self.state_transform(state)
            state_vector = [state, state, state, state]

            done = False

            while not done:
                state = np.expand_dims(np.stack(state_vector, axis=0), axis=0)
                state = torch.Tensor(state).type('torch.FloatTensor').cuda(0)
                action = self.action_transform(self.net.forward(state).squeeze().cpu().detach().numpy())
                new_state, reward, done, info = self.env.step(action)

                gif.append(new_state)

                new_state = self.state_transform(new_state)
                state_vector.pop(0)
                state_vector.append(new_state)

            gif = np.stack(gif, axis=0)
            wandb.log({"video": wandb.Video(gif.swapaxes(3, 2).swapaxes(2, 1)), "epoch": self.current_epoch})
