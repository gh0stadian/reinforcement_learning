import gym
import torch
import wandb


from net import ConvModel
from train import DQNLightning
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from config import train_config, model_config, env_config, action_space
from utils import state_transform, action_transform

AVAIL_GPUS = min(1, torch.cuda.device_count())


wandb.init(project="RL-car", entity="jbdb")
wandb_logger = WandbLogger(log_model=True, config=model_config)

if __name__ == "__main__":
    env = gym.make(env_config['env_name'])

    model = ConvModel(conv_layers=model_config['conv_layers'],
                      num_classes=len(action_space),
                      in_shape=(4, 64, 64),
                      lin_layers=model_config['lin_layers'],
                      )
    wandb_logger.watch(model)

    model = DQNLightning(model=model,
                         env=env,
                         state_transform=state_transform,
                         action_transform=action_transform,
                         **train_config
                         )

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=5000000,
        logger=wandb_logger,
        val_check_interval=100,
    )
    trainer.fit(model)
    wandb.finish()
