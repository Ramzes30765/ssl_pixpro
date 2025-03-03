import timm
import torch
import torch.nn as nn

from src.ppm import PixelPropagationModule
from src.modules import Predictor, Projector


class PixPro(nn.Module):
    def __init__(
        self,
        backbone_name='resnet18',
        pretrained=False,
        projector_blocks=1,
        predictor_blocks=1,
        reduction=4
        ):

        super(PixPro, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        self.in_features = self.backbone(torch.randn(1, 3, 224, 224))[-1].shape[1]
        self.proj_dim = self.in_features * 4
        self.hidden_dim = self.in_features // 4
        
        self.projector = Projector(self.in_features, self.proj_dim, projector_blocks)
        self.predictor = Predictor(self.proj_dim, self.hidden_dim, predictor_blocks)
        self.pixel_propagation = PixelPropagationModule(self.proj_dim, reduction)

    def forward(self, x1, x2):

        f1 = self.backbone(x1)[-1]  # [B, in_features, H, W]
        f2 = self.backbone(x2)[-1]
        
        z1 = self.projector(f1)     # [B, proj_dim, H, W]
        z2 = self.projector(f2)
        
        p1 = self.predictor(z1)     # Предсказания (ветвь, по которой обновляются веса)
        p2 = self.predictor(z2)
        
        y1 = self.pixel_propagation(z1)  # Целевые представления (для target)
        y2 = self.pixel_propagation(z2)
        
        return p1, p2, y1, y2