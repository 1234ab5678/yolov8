import torch
import torch.nn as nn

from nets.Attention import SpatialAttention, ChannelAttention
from nets.backbone import Backbone, C2f, Conv, SiLU, autopad, MobileConv, MobileBottleneck, MobileC2f

class FPN(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        # ------------------------加强特征提取网络------------------------#
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        self.conv3_for_upsample1 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8,
                                       base_depth, shortcut=False)
        # 768, 80, 80 => 256, 80, 80
        self.conv3_for_upsample2 = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth,
                                       shortcut=False)

        # 256, 80, 80 => 256, 40, 40
        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        # 512 + 256, 40, 40 => 512, 40, 40
        self.conv3_for_downsample1 = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth,
                                         shortcut=False)

        # 512, 40, 40 => 512, 20, 20
        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # 1024 * deep_mul + 512, 20, 20 =>  1024 * deep_mul, 20, 20
        self.conv3_for_downsample2 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8,
                                         int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
        # ------------------------加强特征提取网络------------------------#

    def forward(self, feat1, feat2, feat3):
        # ------------------------加强特征提取网络------------------------#
        # 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 40, 40
        P5_upsample = self.upsample(feat3)
        # 1024 * deep_mul, 40, 40 cat 512, 40, 40 => 1024 * deep_mul + 512, 40, 40
        P4 = torch.cat([P5_upsample, feat2], 1)
        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_upsample1(P4)

        # 512, 40, 40 => 512, 80, 80
        P4_upsample = self.upsample(P4)
        # 512, 80, 80 cat 256, 80, 80 => 768, 80, 80
        P3 = torch.cat([P4_upsample, feat1], 1)
        # 768, 80, 80 => 256, 80, 80
        P3 = self.conv3_for_upsample2(P3)

        # 256, 80, 80 => 256, 40, 40
        P3_downsample = self.down_sample1(P3)
        # 512, 40, 40 cat 256, 40, 40 => 768, 40, 40
        P4 = torch.cat([P3_downsample, P4], 1)
        # 768, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_downsample1(P4)

        # 512, 40, 40 => 512, 20, 20
        P4_downsample = self.down_sample2(P4)
        # 512, 20, 20 cat 1024 * deep_mul, 20, 20 => 1024 * deep_mul + 512, 20, 20
        P5 = torch.cat([P4_downsample, feat3], 1)
        # 1024 * deep_mul + 512, 20, 20 => 1024 * deep_mul, 20, 20
        P5 = self.conv3_for_downsample2(P5)
        # ------------------------加强特征提取网络------------------------#

        return P3, P4, P5

class MobileFPN(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        # ------------------------加强特征提取网络------------------------#
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        self.conv3_for_upsample1 = MobileC2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8,
                                       base_depth, shortcut=False)
        # 768, 80, 80 => 256, 80, 80
        self.conv3_for_upsample2 = MobileC2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth,
                                       shortcut=False)

        # 256, 80, 80 => 256, 40, 40
        self.down_sample1 = MobileConv(base_channels * 4, base_channels * 4, 3, 2)
        # 512 + 256, 40, 40 => 512, 40, 40
        self.conv3_for_downsample1 = MobileC2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth,
                                         shortcut=False)

        # 512, 40, 40 => 512, 20, 20
        self.down_sample2 = MobileConv(base_channels * 8, base_channels * 8, 3, 2)
        # 1024 * deep_mul + 512, 20, 20 =>  1024 * deep_mul, 20, 20
        self.conv3_for_downsample2 = MobileC2f(int(base_channels * 16 * deep_mul) + base_channels * 8,
                                         int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
        # ------------------------加强特征提取网络------------------------#

    def forward(self, feat1, feat2, feat3):
        # ------------------------加强特征提取网络------------------------#
        # 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 40, 40
        P5_upsample = self.upsample(feat3)
        # 1024 * deep_mul, 40, 40 cat 512, 40, 40 => 1024 * deep_mul + 512, 40, 40
        P4 = torch.cat([P5_upsample, feat2], 1)
        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_upsample1(P4)

        # 512, 40, 40 => 512, 80, 80
        P4_upsample = self.upsample(P4)
        # 512, 80, 80 cat 256, 80, 80 => 768, 80, 80
        P3 = torch.cat([P4_upsample, feat1], 1)
        # 768, 80, 80 => 256, 80, 80
        P3 = self.conv3_for_upsample2(P3)

        # 256, 80, 80 => 256, 40, 40
        P3_downsample = self.down_sample1(P3)
        # 512, 40, 40 cat 256, 40, 40 => 768, 40, 40
        P4 = torch.cat([P3_downsample, P4], 1)
        # 768, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_downsample1(P4)

        # 512, 40, 40 => 512, 20, 20
        P4_downsample = self.down_sample2(P4)
        # 512, 20, 20 cat 1024 * deep_mul, 20, 20 => 1024 * deep_mul + 512, 20, 20
        P5 = torch.cat([P4_downsample, feat3], 1)
        # 1024 * deep_mul + 512, 20, 20 => 1024 * deep_mul, 20, 20
        P5 = self.conv3_for_downsample2(P5)
        # ------------------------加强特征提取网络------------------------#

        return P3, P4, P5

class AttentionFPN(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()

        self.cam1 = ChannelAttention(base_channels * 8)
        self.cam2 = ChannelAttention(int(base_channels * 16 * deep_mul))
        self.cam3 = ChannelAttention(base_channels * 4)
        self.cam4 = ChannelAttention(base_channels * 8)

        self.sam1 = SpatialAttention(kernel_size=3)
        self.sam2 = SpatialAttention(kernel_size=3)
        self.sam3 = SpatialAttention(kernel_size=3)

        # ------------------------加强特征提取网络------------------------#
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        self.conv3_for_upsample1 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8,
                                       base_depth, shortcut=False)
        # 768, 80, 80 => 256, 80, 80
        self.conv3_for_upsample2 = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth,
                                       shortcut=False)

        # 256, 80, 80 => 256, 40, 40
        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        # 512 + 256, 40, 40 => 512, 40, 40
        self.conv3_for_downsample1 = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth,
                                         shortcut=False)

        # 512, 40, 40 => 512, 20, 20
        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # 1024 * deep_mul + 512, 20, 20 =>  1024 * deep_mul, 20, 20
        self.conv3_for_downsample2 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8,
                                         int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
        # ------------------------加强特征提取网络------------------------#

    def forward(self, feat1, feat2, feat3):
        # ------------------------加强特征提取网络------------------------#
        # 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 40, 40
        feat1 = feat1 * self.sam1(feat1)
        feat2 = feat2 * self.sam2(feat2)
        feat3 = feat3 * self.sam3(feat3)

        P5_upsample = self.upsample(feat3)
        # 1024 * deep_mul, 40, 40 cat 512, 40, 40 => 1024 * deep_mul + 512, 40, 40
        P5_upsample = P5_upsample * self.cam2(P5_upsample)
        P4 = torch.cat([P5_upsample, feat2], 1)
        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_upsample1(P4)

        # 512, 40, 40 => 512, 80, 80
        P4_upsample = self.upsample(P4)
        P4_upsample = P4_upsample * self.cam1(P4_upsample)
        # 512, 80, 80 cat 256, 80, 80 => 768, 80, 80
        P3 = torch.cat([P4_upsample, feat1], 1)
        # 768, 80, 80 => 256, 80, 80
        P3 = self.conv3_for_upsample2(P3)

        # 256, 80, 80 => 256, 40, 40
        P3_downsample = self.down_sample1(P3)
        P3_downsample = P3_downsample * self.cam3(P3_downsample)
        # 512, 40, 40 cat 256, 40, 40 => 768, 40, 40
        P4 = torch.cat([P3_downsample, P4], 1)
        # 768, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_downsample1(P4)

        # 512, 40, 40 => 512, 20, 20
        P4_downsample = self.down_sample2(P4)
        P4_downsample = P4_downsample * self.cam4(P4_downsample)
        # 512, 20, 20 cat 1024 * deep_mul, 20, 20 => 1024 * deep_mul + 512, 20, 20
        P5 = torch.cat([P4_downsample, feat3], 1)
        # 1024 * deep_mul + 512, 20, 20 => 1024 * deep_mul, 20, 20
        P5 = self.conv3_for_downsample2(P5)
        # ------------------------加强特征提取网络------------------------#

        return P3, P4, P5



