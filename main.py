import os, datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from clearml import Task, TaskTypes
from omegaconf import OmegaConf

from src.PixProLightning import PixProModel
from src.datamodule import PixProDataModule
from utils.custom_callbacks import ClusteringVisualizationCallback
from utils.reg_parameters import register_all_parameters, dict_to_omegaconf


def train(cfg_path):

    cfg = OmegaConf.load(cfg_path)

    train_task = Task.init(project_name=cfg.task.proj_name)
    train_task.set_script(
        entry_point="main.py",
        repository="https://github.com/Ramzes30765/ssl_pixpro.git",
        branch="main"
    )

    register_all_parameters(train_task, OmegaConf.to_container(cfg, resolve=True))
    params = train_task.get_parameters_as_dict()['General']
    
    now = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    pretrained = 'pretrained' if params.get('model.pretrained') else 'not_pretrained'
    new_name = f"{params.get('model.backbone')}_{pretrained}_{now}"
    new_tags = [params.get('model.backbone'), params.get('data.dataset_name'), f'{params.get('train.epoch')}_epochs']

    train_task.set_name(new_name)
    train_task.add_tags(new_tags)
    
    cfg = dict_to_omegaconf(params)

    train_task.execute_remotely(queue_name=params.get('pipeline.queue'))

    data_module = PixProDataModule(cfg)
    data_module.setup()
    ssl_model = PixProModel(cfg)

    lr_callback = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.task.proj_name}/{cfg.task.task_name}_{cfg.model.backbone}/{now}/checkpoints",
        filename="epoch{epoch:02d}",
        every_n_epochs=10,
        save_last=True,
        verbose=True
    )
    
    cluster_viz_callback = ClusteringVisualizationCallback(
        val_dataset=data_module.val_dataset,
        eps=cfg.val.eps,
        min_samples=cfg.val.min_samples,
        img_size=cfg.data.img_size,
        output_dir=f"{cfg.task.proj_name}/{cfg.task.task_name}_{cfg.model.backbone}/{now}/cluster_plots"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epoch,
        devices=cfg.train.devices,
        accelerator=cfg.train.accelerator,
        check_val_every_n_epoch=cfg.train.val_step,
        log_every_n_steps=cfg.train.log_step,
        callbacks =[lr_callback, checkpoint_callback, cluster_viz_callback],
        # profiler="simple"
        )

    trainer.fit(ssl_model, datamodule=data_module)

if __name__ == '__main__':
    train('configs/config.yaml')