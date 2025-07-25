import os, datetime

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import PyTorchProfiler
from clearml import Task, TaskTypes

from src.PixProLightning import PixProModel
from src.datamodule import PixProDataModule


def parse_arguments():
    parser = argparse.ArgumentParser(description="SSL PixPro train task")

    # task argiments
    parser.add_argument("--pipeline_pipe_name", type=str, default='SSL pipeline')
    parser.add_argument("--pipeline_proj_name", type=str, default='PixPro')
    parser.add_argument("--pipeline_queue", type=str, default='pixpro_queue')
    parser.add_argument("--task_proj_name", type=str, default='PixPro')
    parser.add_argument("--task_task_name", type=str, default='ResNet')

    # model arguments
    parser.add_argument("--model_backbone", type=str, default='resnet50')
    parser.add_argument("--model_pretrained", action="store_true")
    parser.add_argument("--model_in_features", type=int, default=2048)
    parser.add_argument("--model_proj_dim", type=int, default=256)
    parser.add_argument("--model_hidden_dim", type=int, default=2048)
    parser.add_argument("--model_projector_blocks", type=int, default=1)
    parser.add_argument("--model_predictor_blocks", type=int, default=1)
    parser.add_argument("--model_reduction", type=int, default=4)

    # data arguments
    parser.add_argument("--data_img_size", type=int, default=224)
    parser.add_argument("--data_numclasses", type=int, default=3)
    parser.add_argument("--data_dataset_name", type=str, default='kompressor_coco')
    parser.add_argument("--data_train_folder", type=str, default='train')
    parser.add_argument("--data_val_folder", type=str, default='valid')
    parser.add_argument("--data_train_ann", type=str, default='annotations/instances_train.json')
    parser.add_argument("--data_val_ann", type=str, default='annotations/instances_valid.json')

    # train arguments
    parser.add_argument("--data_batchsize", type=int, default=32)
    parser.add_argument("--data_numworkers", type=int, default=16)
    parser.add_argument("--train_epoch", type=int, default=100)
    parser.add_argument("--train_lr_start", type=float, default=1e-3)
    parser.add_argument("--train_lr_end", type=float, default=1e-5)
    parser.add_argument("--train_devices", type=str, default='auto')
    parser.add_argument("--train_accelerator", type=str, default='auto')
    parser.add_argument("--train_val_step", type=int, default=5)
    parser.add_argument("--train_log_step", type=int, default=5)
    return parser.parse_args()

def register_parameters(task, args):
    for arg in vars(args):
        task.set_parameter(arg, getattr(args, arg))

def train():

    cfg = parse_arguments()
    pretrained = 'pretrained' if cfg.model_pretrained else 'not_pretrained'
    now = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')

    artifacts_dir = "/home/kitt/ssl_pixpro/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    train_task = Task.init(
        project_name=cfg.task_proj_name,
        task_name=cfg.task_task_name,
        task_type=TaskTypes.training,
        output_uri=artifacts_dir
        )

    task_name = f"{cfg.task_task_name}_{now}"
    train_task.set_name(task_name)
    train_task.set_tags([cfg.model_backbone, pretrained, cfg.data_dataset_name, f'{cfg.train_epoch}_epoch'])
    train_task.flush()

    data_module = PixProDataModule(cfg)
    # data_module.setup()
    ssl_model = PixProModel(cfg)

    profiler = PyTorchProfiler(profile_memory=True, dirpath=".", filename="perf_logs")
    lr_callback = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.task_proj_name}/{cfg.task_task_name}_{cfg.model_backbone}/{now}/checkpoints",
        filename="epoch{epoch:02d}",
        every_n_epochs=5,
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train_epoch,
        devices=cfg.train_devices,
        accelerator=cfg.train_accelerator,
        check_val_every_n_epoch=cfg.train_val_step,
        log_every_n_steps=cfg.train_log_step,
        callbacks =[lr_callback, checkpoint_callback],
        profiler=profiler
        )

    trainer.fit(ssl_model, datamodule=data_module)
    profiler.summary()

if __name__ == '__main__':
    train()