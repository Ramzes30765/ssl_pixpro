import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, proj_dim, hidden_dim, num_blocks=1):
        
        super(Predictor, self).__init__()
        self.predictor = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(proj_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, proj_dim, kernel_size=1)
            ) for _ in range(num_blocks)]
        )
    
    def forward(self, x):
        return self.predictor(x)


class Projector(nn.Module):
    def __init__(self, input_dim, proj_dim, num_blocks=1):
        
        super(Projector, self).__init__()
        self.input_conv = nn.Conv2d(input_dim, proj_dim, kernel_size=1)
        self.predictor = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(proj_dim, proj_dim, kernel_size=1),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(proj_dim, proj_dim, kernel_size=1)
            ) for _ in range(num_blocks)]
        )
    
    def forward(self, x):
        x = self.input_conv(x)
        return self.predictor(x)