from models.mlp_head import MLPHead
import torch
from torch import nn
import torch.nn.functional as F
from models.complexcnn import ComplexConv, ComplexConv_trans
import torch.nn as nn
import math
from MobileNetV2 import MobileNetV2

class Encoder_and_projection(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Encoder_and_projection, self).__init__()
        # 输入: [batch, 2, 256]
        self.conv1 = nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        # 残差块序列
        self.resblocks = nn.Sequential(
            ResBlock1D(32, 32),  # 输出: [batch, 32, 128]
            ResBlock1D(32, 32),  # 输出: [batch, 32, 128]
            ResBlock1D(32, 64, first_layer=True),  # 输出: [batch, 64, 128]
            ResBlock1D(64, 64),  # 输出: [batch, 64, 128]
        )
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)  # 输出: [batch, 64, 64]
        self.flatten = nn.Flatten()

        # 计算全连接层输入维度: 64 * 64 = 4096
        self.fc = nn.Linear(64 * 64, 512)
        self.fc_class = nn.Linear(512, 128)
        self.classifier = nn.Linear(128, 6)

        # 修复：使用nn.Module而不是lambda函数
        self.l2_norm = NormalizeLayer()

        self.projetion = MLPHead(in_channels=512, **kwargs['projection_head'])
        self.class_centers = nn.Parameter(
            torch.randn(6, 512),
            requires_grad=True
        )

    def forward(self, x):
        # 初始卷积层
        x = F.relu(self.bn1(self.conv1(x)))  # 输出: [batch, 32, 128]
        # 残差块
        x = self.resblocks(x)  # 输出: [batch, 64, 128]
        # 池化层
        x = self.avgpool(x)  # 输出: [batch, 64, 64]
        # 展平
        x = self.flatten(x)  # 输出: [batch, 4096]
        # 全连接层
        x = F.relu(self.fc(x))  # 输出: [batch, 512]
        # L2特征归一化
        embedding = self.l2_norm(x)  # 输出: [batch, 512]
        project_out = self.projetion(embedding)
        # 分类分支
        y = F.relu(self.fc_class(embedding))  # 输出: [batch, 128]
        logits = self.classifier(y)  # 输出: [batch, num_classes]
        return embedding, project_out, logits, self.class_centers


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, first_layer=False, kernel_size=3):
        super(ResBlock1D, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if first_layer or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)
        return out

    # 新增：创建可序列化的归一化层


class NormalizeLayer(nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)