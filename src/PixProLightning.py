import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
from torchmetrics.classification import MultilabelAveragePrecision
from omegaconf import OmegaConf

from src.pixpro import PixPro
from src.losses import pixpro_loss


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
        
        self.model = PixPro(
            backbone_name=backbone,
            pretrained=pretrained,
            projector_blocks=projector_blocks,
            predictor_blocks=predictor_blocks,
        )
        
        self.num_classes = cfg.data_numclasses
        self.roi_head   = nn.Linear(
            in_features=self.model.proj_dim,
            out_features=self.num_classes
        )
        self.probe = nn.Linear(self.model.proj_dim, self.num_classes)
        self.val_mAP = MultilabelAveragePrecision(num_labels=self.num_classes)
        
        self.max_epoch = self.cfg.train_epoch
        self.lr_start = self.cfg.train_lr_start
        self.lr_end = self.cfg.train_lr_end
        self.img_size = self.cfg.data_img_size
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        view1, view2 = batch
        p1, p2, y1, y2 = self.model(view1, view2)
        loss = pixpro_loss(p1, p2, y1, y2)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        
        with torch.no_grad():
            p, *_ = self.model(imgs, imgs)
            feats = p.mean(dim=(2, 3))
        logits = self.probe(feats)
        
        labels = torch.zeros(imgs.size(0), self.num_classes, device=self.device)
        for i, t in enumerate(targets):
            labels[i, t["labels"]] = 1
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.val_mAP.update(logits.sigmoid(), labels.int())
        self.log_dict(
            {"val_loss": loss, "val_mAP": self.val_mAP},
            prog_bar=True, on_epoch=True, batch_size=imgs.size(0)
        )
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_start)
        scheduler = CosineAnnealingLR(optimizer, T_max=int(self.max_epoch), eta_min=self.lr_end)
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }