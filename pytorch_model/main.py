import gym
import torch
import wandb

from pytorch_model.net_fixed import ConvModel
from trainer import Trainer

from config import train_config, model_config, env_config
from utils import state_transform, action_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')

    model = ConvModel(conv_layers=model_config['conv_layers'],
                      in_shape=(4, 192, 160),
                      lin_layers=model_config['lin_layers'],
                      num_classes=6
                      )

    target_model = ConvModel(conv_layers=model_config['conv_layers'],
                             in_shape=(4, 192, 160),
                             lin_layers=model_config['lin_layers'],
                             num_classes=6
                             )

    trainer = Trainer(model=model,
                      target_model=target_model,
                      env=env,
                      state_transform=state_transform,
                      action_transform=action_transform,
                      device=device,
                      config=train_config
                      )

    trainer.train()
