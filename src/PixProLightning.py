import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
from torchmetrics.classification import MultilabelAveragePrecision

from src.pixpro import PixPro
from src.losses import pixpro_loss


class PixProModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        
        # model params
        self.backbone = self.cfg.model_backbone
        self.pretrained = self.cfg.model_pretrained
        self.in_features = self.cfg.model_in_features
        self.proj_dim = self.cfg.model_proj_dim
        self.hidden_dim = self.cfg.model_hidden_dim
        self.projector_blocks = self.cfg.model_projector_blocks
        self.predictor_blocks = self.cfg.model_predictor_blocks

        self.model = PixPro(
            backbone_name=self.backbone,
            pretrained=self.pretrained,
            in_features=self.in_features,
            proj_dim=self.proj_dim,
            hidden_dim=self.hidden_dim,
            projector_blocks=self.projector_blocks,
            predictor_blocks=self.predictor_blocks,
        )
        
        self.num_classes = cfg.data_numclasses
        self.probe = nn.Linear(self.proj_dim, self.num_classes)
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=view1.size(0))
        
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
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=imgs.size(0))

        # # DEBUG
        # if batch_idx == 0:
        #     print("\n=== VAL BATCH  #0 ===")
        #     print("imgs:",         type(imgs),   imgs.shape)
        #     print("raw targets:",  type(targets), targets)
        #     print("labels:",       labels.shape, labels)
        
        #     print("feats:",        feats.shape, feats)
        #     print("logits:",       logits.shape, logits)
        #     print("sigmoid:",      logits.sigmoid())
        #     print("============================\n")
    
    def on_validation_epoch_end(self):
        val_map = self.val_mAP.compute()
        self.log("val_mAP", val_map, prog_bar=True)
        self.val_mAP.reset()
        
        
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