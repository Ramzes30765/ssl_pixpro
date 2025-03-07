import os

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from clearml import Dataset as ClearmlDataset

from utils.augmentations import base_transforms

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([
            os.path.join(root_dir, file)
            for file in os.listdir(root_dir)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

class PixProDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset_path = ClearmlDataset.get(
            dataset_name=cfg.data_dataset_name
            ).get_local_copy()
        self.train_dir = os.path.join(self.dataset_path, self.cfg.data_train_folder)
        self.val_dir = os.path.join(self.dataset_path, self.cfg.data_val_folder)

        self.batch_size = self.cfg.data_batchsize
        self.num_workers = self.cfg.data_numworkers
        self.img_size = self.cfg.data_img_size

        self.base_transform = base_transforms(self.img_size)

    def setup(self, stage=None):
        self.train_dataset = ImageFolderDataset(root_dir=self.train_dir, transform=self.base_transform)
        self.val_dataset   = ImageFolderDataset(root_dir=self.val_dir, transform=self.base_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)
