import os, datetime

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import PyTorchProfiler
from clearml import Task, TaskTypes

from src.PixProLightning import PixProModel
from src.datamodule import PixProDataModule
from utils.custom_callbacks import ClusteringVisualizationCallback


def parse_arguments():
    parser = argparse.ArgumentParser(description="SSL PixPro train task")
    parser.add_argument("--pipeline_pipe_name", type=str, default='SSL pipeline', help="Название пайплайна")
    parser.add_argument("--pipeline_proj_name", type=str, default='PixPro', help="Íàçâàíèå ïðîåêòà ïàéïëàéíà")
    parser.add_argument("--pipeline_queue", type=str, default='pixpro_queue', help="Î÷åðåäü äëÿ âûïîëíåíèÿ")
    parser.add_argument("--task_proj_name", type=str, default='PixPro', help="Íàçâàíèå ïðîåêòà çàäà÷è")
    parser.add_argument("--task_task_name", type=str, default='ResNet', help="Íàçâàíèå çàäà÷è")
    parser.add_argument("--model_backbone", type=str, default='resnet18', help="Àðõèòåêòóðà ìîäåëè")
    parser.add_argument("--model_pretrained", type=bool, default=False, help="Èñïîëüçîâàòü ïðåäîáó÷åííóþ ìîäåëü")
    parser.add_argument("--model_projector_blocks", type=int, default=1, help="Êîëè÷åñòâî áëîêîâ ïðîåêòîðà")
    parser.add_argument("--model_predictor_blocks", type=int, default=1, help="Êîëè÷åñòâî áëîêîâ ïðåäèêòîðà")
    parser.add_argument("--model_reduction", type=int, default=4, help="Êîýôôèöèåíò óìåíüøåíèÿ")
    parser.add_argument("--data_img_size", type=int, default=640, help="Ðàçìåð èçîáðàæåíèÿ")
    parser.add_argument("--data_dataset_name", type=str, default='ssl_turbine_dataset', help="Íàçâàíèå íàáîðà äàííûõ")
    parser.add_argument("--data_train_folder", type=str, default='turbine_train', help="Ïàïêà ñ îáó÷àþùèìè äàííûìè")
    parser.add_argument("--data_val_folder", type=str, default='turbine_val', help="Ïàïêà ñ âàëèäàöèîííûìè äàííûìè")
    parser.add_argument("--data_batchsize", type=int, default=32, help="Ðàçìåð áàò÷à")
    parser.add_argument("--data_numworkers", type=int, default=16, help="Êîëè÷åñòâî ïîòîêîâ çàãðóçêè äàííûõ")
    parser.add_argument("--train_epoch", type=int, default=5, help="Êîëè÷åñòâî ýïîõ")
    parser.add_argument("--train_lr_start", type=float, default=1e-3, help="Íà÷àëüíàÿ ñêîðîñòü îáó÷åíèÿ")
    parser.add_argument("--train_lr_end", type=float, default=1e-5, help="Êîíå÷íàÿ ñêîðîñòü îáó÷åíèÿ")
    parser.add_argument("--train_devices", type=str, default='auto', help="Óñòðîéñòâà äëÿ îáó÷åíèÿ")
    parser.add_argument("--train_accelerator", type=str, default='auto', help="Òèï àêñåëåðàöèè")
    parser.add_argument("--train_val_step", type=int, default=10, help="Øàã âàëèäàöèè")
    parser.add_argument("--train_log_step", type=int, default=5, help="Øàã ëîãèðîâàíèÿ")
    parser.add_argument("--val_eps", type=float, default=0.5, help="Ïàðàìåòð eps äëÿ DBSCAN")
    parser.add_argument("--val_min_samples", type=int, default=5, help="Ìèíèìàëüíîå êîëè÷åñòâî îáðàçöîâ äëÿ êëàñòåðà")
    parser.add_argument("--val_sample_fraction", type=float, default=1.0, help="Äîëÿ âûáîðêè äëÿ âàëèäàöèè")
    return parser.parse_args()

def register_parameters(task, args):
    for arg in vars(args):
        task.set_parameter(arg, getattr(args, arg))

def train():

    cfg = parse_arguments()
    pretrained = 'pretrained' if cfg.model_pretrained else 'not_pretrained'
    now = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')


    train_task = Task.init(
        project_name=cfg.task_proj_name,
        task_name=cfg.task_task_name,
        task_type=TaskTypes.training,
        output_uri='/home/kitt/ssl_pixpro/s3_demo'
        )

    task_name = f"{cfg.task_task_name}_{now}"
    train_task.set_name(task_name)
    train_task.set_tags([cfg.model_backbone, pretrained, cfg.data_dataset_name, f'{cfg.train_epoch}_epoch'])
    train_task.flush()

    train_task.execute_remotely(queue_name=cfg.pipeline_queue)

    data_module = PixProDataModule(cfg)
    data_module.setup()
    ssl_model = PixProModel(cfg)

    profiler = PyTorchProfiler(profile_memory=True, dirpath=".", filename="perf_logs")
    lr_callback = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.task_proj_name}/{cfg.task_task_name}_{cfg.model_backbone}/{now}/checkpoints",
        filename="epoch{epoch:02d}",
        every_n_epochs=5,
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