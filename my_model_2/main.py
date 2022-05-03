import gym
import torch
import wandb

from my_model_2.models.net_classification import ConvModel
from train import DQNLightning
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from config import train_config, model_config, env_config
from utils import state_transform, action_transform

AVAIL_GPUS = min(1, torch.cuda.device_count())

wandb.init(project="RL-car", entity="jbdb")
wandb_logger = WandbLogger(config=model_config)

if __name__ == "__main__":
    env = gym.make(env_config['env_name'])

    model = ConvModel(conv_layers=model_config['conv_layers'],
                      in_shape=(4, 64, 64),
                      lin_layers=model_config['lin_layers'],
                      )

    wandb_logger.watch(model, log="all")

    target_model = ConvModel(conv_layers=model_config['conv_layers'],
                             in_shape=(4, 64, 64),
                             lin_layers=model_config['lin_layers'],
                             )

    model = DQNLightning(model=model,
                         target_model=target_model,
                         env=env,
                         state_transform=state_transform,
                         action_transform=action_transform,
                         **train_config
                         )

    checkpoint_callback = ModelCheckpoint(monitor="total_reward",
                                          mode="max",
                                          dirpath='checkpoints/',
                                          filename='sample-mnist-{epoch:02d}-{total_reward:.2f}',
                                          save_top_k=2
                                          )

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=5000000,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=100,
    )

    trainer.fit(model)
    wandb.finish()
