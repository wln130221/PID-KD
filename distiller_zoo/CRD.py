import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Normalize(nn.Module):
    """ L2 normalization layer """
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(dim=1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm + 1e-8)
        return out


class Embed(nn.Module):
    """Linear embedding + L2 normalize"""
    def __init__(self, dim_in, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten to [B, D]
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class CRD(nn.Module):
    def __init__(self, s_dim, t_dim, feat_dim=128):
        super(CRD, self).__init__()
        self.embed_s = Embed(s_dim, feat_dim)
        self.embed_t = Embed(t_dim, feat_dim)
        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(self, feat_s, feat_t):
        """
        feat_s: student features, shape [B, C, H, W] or [B, D]
        feat_t: teacher features, shape [B, C, H, W] or [B, D]
        """
        # embed and normalize
        feat_s = self.embed_s(feat_s)  # [B, feat_dim]
        feat_t = self.embed_t(feat_t)  # [B, feat_dim]
        target = torch.ones(feat_s.size(0)).to(feat_s.device)  # positive pairs
        loss = self.loss_fn(feat_s, feat_t, target)
        return loss
