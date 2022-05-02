import gym
import torch
import wandb

from pytorch_model.net_fixed import ConvModel
from trainer import Trainer

from config import train_config, model_config, env_config, action_space
from utils import state_transform, action_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    env = gym.make(env_config['env_name'])

    model = ConvModel(conv_layers=model_config['conv_layers'],
                      in_shape=(4, 64, 64),
                      lin_layers=model_config['lin_layers'],
                      num_classes=len(action_space)
                      )

    target_model = ConvModel(conv_layers=model_config['conv_layers'],
                             in_shape=(4, 64, 64),
                             lin_layers=model_config['lin_layers'],
                             num_classes=len(action_space)
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
