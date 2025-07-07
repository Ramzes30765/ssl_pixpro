import torch.nn as nn

def _init_conv(m: nn.Module):
    """Kaiming init conv weights as in original PixPro implementation."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class Projector(nn.Module):
    def __init__(self, input_dim, proj_dim, num_blocks=1):
        
        super(Projector, self).__init__()
        self.input_conv = nn.Conv2d(input_dim, proj_dim, use_relu=False, kernel_size=1)
        if self.use_relu:
            self.projector = nn.Sequential(
                *[nn.Sequential(
                    nn.Conv2d(proj_dim, proj_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(proj_dim, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(proj_dim, proj_dim, kernel_size=1, bias=False)
                ) for _ in range(num_blocks)]
            )
        else:
            self.projector = nn.Sequential(
                *[nn.Sequential(
                    nn.Conv2d(proj_dim, proj_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(proj_dim, affine=True)
                ) for _ in range(num_blocks)]
            )
        self.apply(_init_conv)
    def forward(self, x):
        x = self.input_conv(x)
        return self.projector(x)
    
    
class Predictor(nn.Module):
    def __init__(self, proj_dim, hidden_dim, num_blocks=1):
        
        super(Predictor, self).__init__()
        self.predictor = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(proj_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, proj_dim, kernel_size=1, bias=False)
            ) for _ in range(num_blocks)]
        )
        self.apply(_init_conv)
    def forward(self, x):
        return self.predictor(x)