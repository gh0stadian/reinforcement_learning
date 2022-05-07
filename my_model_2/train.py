import gym
import torch
import wandb
import math
import pytorch_lightning as plz

from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from agent import Agent
from dataset import RLDataset, ReplayBuffer
from utils import action_transform, state_transform
from my_model_2.net import ConvModel


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
            self,
            model_config: dict,
            env_config: dict,
            wrappers_config=None,
            batch_size: int = 16,
            lr: float = 1e-2,
            gamma: float = 0.99,
            sync_rate: int = 10,
            reward_decreasing_limit: int = 10,
            replay_size: int = 1000,
            warm_start_size: int = 1000,
            eps_last_frame: int = 1000,
            eps_start: float = 1.0,
            log_video_episode: int = 20,
            eps_end: float = 0.01,
            eps_decay: float = 200,
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
        self.save_hyperparameters()
        self.state_transform = state_transform
        self.action_transform = action_transform
        self.buffer = ReplayBuffer(self.hparams.replay_size)

        self.env = gym.make(self.hparams.env_config['env_name'], render_mode='rgb_array')

        self.net = ConvModel(conv_layers=self.hparams.model_config['conv_layers'],
                             in_shape=(4, 192, 160),
                             lin_layers=self.hparams.model_config['lin_layers'],
                             num_classes=6
                             )

        wandb.watch(self.net, log="all")

        self.target_net = ConvModel(conv_layers=self.hparams.model_config['conv_layers'],
                                    in_shape=(4, 192, 160),
                                    lin_layers=self.hparams.model_config['lin_layers'],
                                    num_classes=6
                                    )

        self.agent = Agent(self.env,
                           self.buffer,
                           state_transform=state_transform,
                           wrappers_config=self.hparams.wrappers_config
                           )

        self.populate(self.hparams.warm_start_steps)
        self.total_reward = 0
        self.episode_reward = 0
        self.episodes = 0

    def populate(self, steps=1000):
        print("Populating buffer")
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x):
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch, device):
        states, actions, rewards, dones, next_states = batch
        ones_vector = torch.ones(len(dones))
        gamma = torch.FloatTensor([self.hparams.gamma] * len(dones))

        if device not in ["cpu"]:
            dones = dones.cuda(device)
            states = states.cuda(device)
            next_states = next_states.cuda(device)
            ones_vector = ones_vector.cuda(device)
            actions = actions.cuda(device)
            gamma = gamma.cuda(device)

        state_action_values = self.net(states)
        state_action_values = state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states)
            next_state_values = next_state_values.max(1)[0]

        dones = (ones_vector - dones)
        expected_state_action_values = (next_state_values * gamma * dones) + rewards

        # penalty = self.calc_not_moving_penalty(states)
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch, nb_batch):
        device = self.get_device(batch)

        epsilon = self.hparams.eps_end + (self.hparams.eps_start - self.hparams.eps_end) * \
                  math.exp(-1. * self.episodes / self.hparams.eps_decay)

        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward
        loss = self.dqn_mse_loss(batch, device)

        if done:
            self.total_reward = self.episode_reward
            self.episodes += 1
            self.episode_reward = 0

            if (self.episodes+1) % self.hparams.log_video_episode == 0:
                video = self.agent.record_play(self.net, self.device)
                wandb.log({"video": wandb.Video(video.swapaxes(3, 2).swapaxes(2, 1), fps=24)}, step=self.episodes)

        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log("train/reward", self.total_reward)
        self.log("train/episodes", self.episodes)
        self.log("train/loss", loss)
        return loss

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