import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np
import math

from ._base import Distiller


def feat_loss(source, target, margin):
    margin = margin.to(source)
    loss = (
        (source - margin) ** 2 * ((source > margin) & (target <= margin)).float()
        + (source - target) ** 2
        * ((source > target) & (target > margin) & (target <= 0)).float()
        + (source - target) ** 2 * (target > 0).float()
    )
    return torch.abs(loss).mean(dim=0).sum()


class ConnectorConvBN(nn.Module):
    def __init__(self, s_channels, t_channels, kernel_size=1):
        super(ConnectorConvBN, self).__init__()
        self.s_channels = s_channels
        self.t_channels = t_channels
        self.connectors = nn.ModuleList(
            self._make_conenctors(s_channels, t_channels, kernel_size)
        )

    def _make_conenctors(self, s_channels, t_channels, kernel_size):
        assert len(s_channels) == len(t_channels), "unequal length of feat list"
        connectors = nn.ModuleList(
            [
                self._build_feature_connector(t, s, kernel_size)
                for t, s in zip(t_channels, s_channels)
            ]
        )
        return connectors

    def _build_feature_connector(self, t_channel, s_channel, kernel_size):
        C = [
            nn.Conv2d(
                s_channel,
                t_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(t_channel),
        ]
        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return nn.Sequential(*C)

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class OFD(nn.Module):
    def __init__(self, ce_weight=1.0, feat_weight=1.0, kernel_size=1):
        super(OFD, self).__init__()
        self.ce_loss_weight = ce_weight
        self.feat_loss_weight = feat_weight
        self.kernel_size = kernel_size

        self.connectors = None
        self.margins = None
        self.initialized = False

    def forward_train(self, image, target, student, teacher):
        logits_student, feature_student = student(image)
        with torch.no_grad():
            _, feature_teacher = teacher(image)

        # 第一次 forward 先初始化 connector 和 margin
        if not self.initialized:
            self._init_connectors_and_margins(feature_student, feature_teacher, device=image.device)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_feat = self.feat_loss_weight * self.ofd_loss(
            self.connectors(feature_student["preact_feats"][1:]),
            feature_teacher["preact_feats"][1:]
        )
        losses_dict = {"loss_ce": loss_ce, "loss_kd": loss_feat}
        return logits_student, losses_dict

    def _init_connectors_and_margins(self, feature_student, feature_teacher, device):
        # 初始化 connector
        s_feats = feature_student["preact_feats"][1:]
        t_feats = feature_teacher["preact_feats"][1:]

        s_channels = [f.shape[1] for f in s_feats]
        t_channels = [f.shape[1] for f in t_feats]
        self.connectors = ConnectorConvBN(s_channels, t_channels, kernel_size=self.kernel_size).to(device)

        # 初始化 margins
        self.margins = []
        for t_feat in t_feats:
            B, C, H, W = t_feat.shape
            t_flat = t_feat.permute(1, 0, 2, 3).reshape(C, -1)
            std = t_flat.std(dim=1)
            mean = t_flat.mean(dim=1)

            margin = []
            for s, m in zip(std, mean):
                s = abs(s.item())
                m = m.item()
                if norm.cdf(-m / s) > 0.001:
                    val = (
                        -s * math.exp(-((m / s) ** 2) / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m
                    )
                else:
                    val = -3 * s
                margin.append(val)
            margin = torch.tensor(margin, dtype=torch.float32, device=device)
            self.margins.append(margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.initialized = True

    def forward(self, f_s, f_t):
        if self.connectors is None or self.margins is None:
            # ✅ 强制将通道数转换为 int，避免 tensor 被误传入 Conv2d
            s_channels = [int(f.shape[1]) for f in f_s]
            t_channels = [int(f.shape[1]) for f in f_t]
            self.connectors = ConnectorConvBN(s_channels, t_channels, kernel_size=self.kernel_size).to(f_s[0].device)

            self.margins = []
            for t_feat in f_t:
                B, C, H, W = t_feat.shape
                t_flat = t_feat.permute(1, 0, 2, 3).reshape(C, -1)
                std = t_flat.std(dim=1)
                mean = t_flat.mean(dim=1)

                margin = []
                for s, m in zip(std, mean):
                    s = abs(s.item())
                    m = m.item()
                    if norm.cdf(-m / s) > 0.001:
                        val = (
                                -s * math.exp(-((m / s) ** 2) / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m
                        )
                    else:
                        val = -3 * s
                    margin.append(val)
                margin = torch.tensor(margin, dtype=torch.float32, device=f_s[0].device)
                self.margins.append(margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        # 做特征对齐并计算loss
        student_feats = self.connectors(f_s)
        teacher_feats = f_t
        return self.ofd_loss(student_feats, teacher_feats)

    def ofd_loss(self, student_feats, teacher_feats):
        loss = 0
        feat_num = len(student_feats)
        for i in range(feat_num):
            s_feat = student_feats[i]
            t_feat = F.adaptive_avg_pool2d(teacher_feats[i], s_feat.shape[-2:]).detach()
            margin = self.margins[i]
            loss += feat_loss(s_feat, t_feat, margin) / (2 ** (feat_num - i - 1))
        return loss
