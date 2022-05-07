import torch
import wandb
from datetime import datetime
from train import DQNLightning
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from config import train_config, model_config, env_config, wrappers_config


class CheckpointEveryEpoch(pl.Callback):
    def __init__(self, log_every_n_epoch, save_path):
        self.log_every_n_epoch = log_every_n_epoch
        self.file_path = save_path

    def on_epoch_end(self, trainer: pl.Trainer, _):
        epoch = trainer.current_epoch
        if (epoch+1) % self.log_every_n_epoch == 0:
            ckpt_path = f"{self.file_path}.ckpt"
            trainer.save_checkpoint(ckpt_path)


AVAIL_GPUS = min(1, torch.cuda.device_count())

wandb.init(project="RL-car", entity="jbdb")


timestamp = datetime.now().strftime("%m_%d_%Y_T_%H_%M_%S")
path = f"checkpoints/{timestamp}"

wandb_logger = WandbLogger(config={'train': train_config,
                                   'model': model_config,
                                   'env': env_config,
                                   'wrappers': wrappers_config,
                                   'checkpoint_path': path,
                                   }
                           )

if __name__ == "__main__":
    model = DQNLightning(model_config=model_config,
                         env_config=env_config,
                         wrappers_config=wrappers_config,
                         **train_config
                         )
    # model = DQNLightning.load_from_checkpoint(checkpoint_path="checkpoints/05_04_2022_T_16_12_58/_e35.ckpt")
    wandb_logger.watch(model, log="all")

    checkpoint_callback = CheckpointEveryEpoch(2000, path)

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=50000000000000,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=100,
    )

    trainer.fit(model)
    wandb.finish()
