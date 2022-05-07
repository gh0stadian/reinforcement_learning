import gym
import torch
import wandb
import copy


from pytorch_model.net_fixed import ConvModel
from trainer import Trainer

from config import train_config, model_config, env_config, load_model
from utils import state_transform, action_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    env = gym.make('SpaceInvaders-v4', render_mode='rgb_array')
    episode = 0

    model = ConvModel(conv_layers=model_config['conv_layers'],
                      in_shape=(4, 192, 160),
                      lin_layers=model_config['lin_layers'],
                      num_classes=6
                      ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])

    if "path" in load_model:
        print("loading model from checkpoint....")
        checkpoint = torch.load(load_model['path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode = checkpoint['episode']

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      env=env,
                      state_transform=state_transform,
                      action_transform=action_transform,
                      device=device,
                      config=train_config,
                      episode=episode
                      )

    trainer.train()
