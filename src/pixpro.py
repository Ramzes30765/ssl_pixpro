import timm
import torch
import torch.nn as nn

from src.ppm import LocalPPM
from src.modules import Predictor, Projector


class PixPro(nn.Module):
    def __init__(
        self,
        backbone_name='resnet18',
        pretrained=False,
        in_features=256,
        proj_dim=256,
        hidden_dim=64,
        projector_blocks=1,
        predictor_blocks=1
        ):

        super(PixPro, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        self.projector = Projector(in_features, proj_dim, projector_blocks)
        self.predictor = Predictor(proj_dim, hidden_dim, predictor_blocks)
        self.pixel_propagation = LocalPPM()
        
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval() 

    def forward(self, x1, x2):

        f1 = self.backbone(x1)[-1]
        f2 = self.backbone(x2)[-1]
        
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        y1 = self.pixel_propagation(z1)
        y2 = self.pixel_propagation(z2)
        
        return p1, p2, y1.detach(), y2.detach()