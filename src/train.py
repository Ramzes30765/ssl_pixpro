import os, datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from clearml import Task, TaskTypes
from omegaconf import OmegaConf

from src.PixProLightning import PixProModel
from src.datamodule import PixProDataModule


def train(cfg_path):

    cfg = OmegaConf.load(cfg_path)
    
    data_module = PixProDataModule(cfg)
    data_module.setup()
    ssl_model = PixProModel(cfg)

    now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    pretrained = 'pretrain' if cfg.model.pretrained else 'nopretrain'

    train_task = Task.init(
        project_name=cfg.task.proj_name,
        task_name=f"{cfg.task.task_name}_{cfg.model.backbone}_{pretrained}_{now}",
        task_type=TaskTypes.training,
        tags=[cfg.model.backbone, pretrained, f'epochs-{cfg.train.epoch}', cfg.data.dataset_name],
        )
    train_task.connect_configuration(cfg_path)

    lr_callback = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.task.proj_name}/{cfg.task.task_name}_{cfg.model.backbone}/{now}/checkpoints",
        filename="epoch{epoch:02d}",
        every_n_epochs=10,
        save_last=True,
        verbose=True
    )
    trainer = pl.Trainer(
        max_epochs=cfg.train.epoch,
        devices=cfg.train.devices,
        accelerator=cfg.train.accelerator,
        check_val_every_n_epoch=cfg.train.val_step,
        log_every_n_steps=cfg.train.log_step,
        callbacks =[lr_callback, checkpoint_callback],
        )
    trainer.fit(ssl_model, datamodule=data_module)

if __name__ == '__main__':
    train('configs/config.yaml')