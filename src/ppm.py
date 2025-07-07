import torch
import torch.nn as nn
from torch.nn.attention import flex_attention
import torch.nn.functional as F

def _init_gamma(m: nn.Module, value: float = 0.2):
    for p in m.parameters():
        if p.requires_grad and p.ndim == 0:
            p.data.fill_(value)
            

class LocalPPM(nn.Module):
    def __init__(
        self,
        radius: int = 2,
        topk: int = 10,
        tau: float = 0.1,
        lam: float = 0.2,
        detach_target: bool = True,
    ):
        super().__init__()
        self.r = radius
        self.k = topk
        self.tau = tau
        self.gamma = nn.Parameter(torch.full((), lam))
        self.detach_target = detach_target
        _init_gamma(self, lam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → y: [B, C, H, W]"""
        if self.detach_target and x.requires_grad:
            x_detached = x.detach()
        else:
            x_detached = x

        B, C, H, W = x_detached.shape
        ksize = 2 * self.r + 1
        patches = F.unfold(x_detached, kernel_size=ksize, padding=self.r)
        patches = patches.view(B, C, ksize * ksize, H, W)
        center = x_detached.unsqueeze(2)

        sims = F.cosine_similarity(patches, center, dim=1) / self.tau

        if self.k < patches.size(2):
            topk_val, topk_idx = sims.topk(self.k, dim=1)
            mask = torch.full_like(sims, float("-inf"))
            mask.scatter_(1, topk_idx, topk_val)
            sims = mask
        weights = F.softmax(sims, dim=1)
        y = (patches * weights.unsqueeze(1)).sum(2)

        return x + self.gamma * y


class FlexAttentionPPM(nn.Module):
    def __init__(self, in_channels, dropout_p=0.0, reduction=4, is_causal=False):

        super(FlexAttentionPPM, self).__init__()
        self.qkv_proj = nn.Conv2d(in_channels, in_channels * 3 // reduction, kernel_size=1)
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        B, C, H, W = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)
        
        q = q.view(B, C, -1).permute(0, 2, 1)
        k = k.view(B, C, -1).permute(0, 2, 1)
        v = v.view(B, C, -1).permute(0, 2, 1)
        
        attn_out = flex_attention(query=q, key=k, value=v, 
                                  attn_mask=None,
                                  dropout_p=self.dropout_p,
                                  is_causal=self.is_causal)

        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        out = self.gamma * attn_out + x
        return out