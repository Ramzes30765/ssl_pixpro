import os
from pathlib import Path

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from clearml import Dataset as ClearmlDataset

from src.augmentations  import SSLPairTransform
from src.dataset    import CocoImagePairDataset, CocoDetDataset

class PixProDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = ClearmlDataset.get(dataset_name=cfg.data_dataset_name, alias=cfg.data_dataset_name)
        self.dataset_root = Path(self.dataset.get_mutable_local_copy('dataset'))
        self.train_imgs = self.dataset_root / cfg.data_train_folder
        self.val_imgs   = self.dataset_root / cfg.data_val_folder
        self.train_ann  = self.dataset_root / cfg.data_train_ann
        self.val_ann    = self.dataset_root / cfg.data_val_ann

        self.batch_size  = cfg.data_batchsize
        self.num_workers = cfg.data_numworkers
        self.img_size    = cfg.data_img_size
        
        self.pair_tf = SSLPairTransform(self.img_size)
    
    def setup(self, stage: str | None = None):
        self.train_dataset = CocoImagePairDataset(
            images_dir=self.train_imgs,
            ann_file=self.train_ann,
            transform_pair=self.pair_tf
        )
        self.val_dataset = CocoDetDataset(
            images_dir=self.val_imgs,
            ann_file=self.val_ann,
            img_size=self.img_size
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=CocoDetDataset.detection_collate
        )
