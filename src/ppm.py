import torch
import torch.nn as nn
from torch.nn.attention import flex_attention
import torch.nn.functional as F


class PixelPropagationModule(nn.Module):
    def __init__(self, in_channels, reduction=4):

        super(PixelPropagationModule, self).__init__()
        self.inter_channels = in_channels // reduction
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, self.inter_channels, -1).permute(0, 2, 1)  # [B, H*W, C]
        proj_key   = self.key_conv(x).view(B, self.inter_channels, -1) # [B, C, H*W]
        score = torch.bmm(proj_query, proj_key) # [B, H*W, H*W]
        attention = F.softmax(score, dim=-1) # [B, H*W, H*W]
        proj_value = self.value_conv(x).view(B, C, -1) # [B, C, H*W]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # transpose attention - [B, C, H*W]
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out


class FlexAttentionPPM(nn.Module):
    def __init__(self, in_channels, dropout_p=0.0, is_causal=False):

        super(FlexAttentionPPM, self).__init__()
        # Проекция для формирования Q, K, V в один шаг
        self.qkv_proj = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        # Обучаемый коэффициент для остаточного соединения
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        B, C, H, W = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)  # каждый [B, C, H, W]
        
        q = q.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        k = k.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        v = v.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        attn_out = flex_attention(query=q, key=k, value=v, 
                                  attn_mask=None,
                                  dropout_p=self.dropout_p,
                                  is_causal=self.is_causal)

        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        out = self.gamma * attn_out + x
        return out