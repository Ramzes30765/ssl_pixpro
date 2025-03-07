import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from omegaconf import OmegaConf

from src.pixpro import PixPro
from src.losses import pixpro_loss
from src.clustering_val import global_clustering_dbscan, per_image_clustering_dbscan
from utils.augmentations import batch_augmentations


class PixProModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        
        # model params
        backbone = self.cfg.model_backbone
        pretrained = self.cfg.model_pretrained
        projector_blocks = self.cfg.model_projector_blocks
        predictor_blocks = self.cfg.model_predictor_blocks
        reduction = self.cfg.model_reduction
        
        self.model = PixPro(
            backbone_name=backbone,
            pretrained=pretrained,
            projector_blocks=projector_blocks,
            predictor_blocks=predictor_blocks,
            reduction=reduction
        )
        
        # train params
        self.epoch = self.cfg.train_epoch
        self.lr_start = self.cfg.train_lr_start
        self.lr_end = self.cfg.train_lr_end
        
        # val params
        self.eps = self.cfg.val_eps
        self.min_samples = self.cfg.val_min_samples
        self.sample_fraction = self.cfg.val_sample_fraction

        # data params
        self.img_size = self.cfg.data_img_size
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, _ = batch
        
        x1 = batch_augmentations(images.clone(), self.img_size)
        x2 = batch_augmentations(images.clone(), self.img_size)
        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        p1, p2, y1, y2 = self.model(x1, x2)
        loss = pixpro_loss(p1, p2, y1, y2)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, outputs):
        
        # Глобальная кластеризация
        self.val_dataloader = self.trainer.val_dataloaders
        global_results = global_clustering_dbscan(self.model, self.val_dataloader, self.device,
                                                    eps=self.eps, min_samples=self.min_samples, 
                                                    sample_fraction=self.sample_fraction)
        if global_results[0] is not None:
            sil, db_index, ch_score, _ = global_results
            val_global_sil = sil
            val_global_db = db_index
            val_global_ch = ch_score
        else:
            val_global_sil = -1
            val_global_db = -1
            val_global_ch = -1

        # Кластеризация по отдельности для каждого изображения
        per_img_results = per_image_clustering_dbscan(self.model, self.val_dataloader, self.device,
                                                        eps=self.eps, min_samples=self.min_samples)
        if per_img_results[0] is not None:
            avg_sil, avg_db, avg_ch = per_img_results
            val_perimg_sil = avg_sil
            val_perimg_db = avg_db
            val_perimg_ch = avg_ch
        else:
            val_perimg_sil = -1
            val_perimg_db = -1
            val_perimg_ch = -1

        # Логируем каждую метрику один раз
        self.log('val_global_sil', val_global_sil, prog_bar=True)
        self.log('val_global_db', val_global_db, prog_bar=True)
        self.log('val_global_ch', val_global_ch, prog_bar=True)
        self.log('val_perimg_sil', val_perimg_sil, prog_bar=True)
        self.log('val_perimg_db', val_perimg_db, prog_bar=True)
        self.log('val_perimg_ch', val_perimg_ch, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_start)
        scheduler = CosineAnnealingLR(optimizer, T_max=int(self.epoch), eta_min=self.lr_end)
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }