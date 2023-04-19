import math

import numpy as np
import torch
import torch.nn as nn

from nets.backbone import Backbone, C2f, Conv, SiLU, autopad, MobileConv, MobileBottleneck, MobileC2f
from nets.convnext import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
from nets.densenet import densenet121, densenet161, densenet169, densenet201
from nets.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from nets.repvgg import create_RepVGG_A0, create_RepVGG_A1, create_RepVGG_A2, create_RepVGG_B0, create_RepVGG_B1, create_RepVGG_B2, create_RepVGG_B3, create_RepVGG_B1g2, create_RepVGG_B1g4, create_RepVGG_B2g2, create_RepVGG_B2g4, create_RepVGG_B3g2, create_RepVGG_B3g4, create_RepVGG_D2se
from nets.mobilenetv2 import mobilenet_v2
from nets.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from nets.MobileViT.model import mobile_vit_xx_small, mobile_vit_x_small, mobile_vit_small
from nets.DeformableCSPDarkNet.deformable_cspdarknet import DeformableCSPDarkNet
from nets.MobileCSPDarkNet.mobile_cspdarknet import MobileCSPDarkNet
from nets.SwinTransformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224, swin_base_patch4_window12_384, swin_base_patch4_window7_224_in22k, swin_base_patch4_window12_384_in22k, swin_large_patch4_window7_224_in22k,swin_large_patch4_window12_384_in22k
from nets.neck import FPN, MobileFPN, AttentionFPN
from nets.FasterNet.model import fasternet_s, fasternet_m, fasternet_l
from nets.resnet import resnet34, resnet50, resnet101, resnet18, resnet152
from utils.utils_bbox import make_anchors

def fuse_conv_and_bn(conv, bn):
    # 混合Conv2d + BatchNorm2d 减少计算量
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备kernel
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class DFL(nn.Module):
    # DFL模块
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1     = c1

    def forward(self, x):
        # bs, self.reg_max * 4, 8400
        b, c, a = x.shape
        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
        
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi, backbone='cspdarknet', neck='FPN', pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
        width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是3, 640, 640
        #-----------------------------------------------#

        #---------------------------------------------------#   
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   256, 80, 80
        #   512, 40, 40
        #   1024 * deep_mul, 20, 20
        #---------------------------------------------------#
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#
        self.backbone_name = backbone
        self.neck_name = neck
        #if backbone == "cspdarknet" or backbone == "DeformableCSPDarkNet":
        if backbone in {"cspdarknet", "DeformableCSPDarkNet", "MobileCSPDarkNet"}:
            # ---------------------------------------------------#
            #   生成CSPdarknet53的主干模型
            #   获得三个有效特征层，他们的shape分别是：
            #   80,80,256
            #   40,40,512
            #   20,20,1024
            # ---------------------------------------------------#
            #self.backbone = Backbone(base_channels, base_depth, deep_mul, phi, pretrained)
            self.backbone = {
                'cspdarknet': Backbone,
                'DeformableCSPDarkNet': DeformableCSPDarkNet,
                'MobileCSPDarkNet': MobileCSPDarkNet,
            }[backbone](base_channels, base_depth, deep_mul, phi, pretrained)
        elif backbone in {"fasternet_s", "fasternet_l", "fasternet_m"}:
            # ---------------------------------------------------#
            #   如果输入不为cspdarknet，则调整通道数
            #   使其符合YoloV5的格式
            # ---------------------------------------------------#
            self.backbone = {
                'fasternet_s': fasternet_s,
                'fasternet_l': fasternet_l,
                'fasternet_m': fasternet_m,
            }[backbone]()
            in_channels = {
                'fasternet_s': [256, 512, 1024],
                'fasternet_m': [288, 576, 1152],
                'fasternet_l': [384, 768, 1536]
            }[backbone]
            feat1_c, feat2_c, feat3_c = in_channels
            self.conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
            self.conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            self.conv_1x1_feat3 = Conv(feat3_c, int(base_channels * 16 * deep_mul), 1, 1)
        else:
            # ---------------------------------------------------#
            #   如果输入不为cspdarknet，则调整通道数
            #   使其符合YoloV5的格式
            # ---------------------------------------------------#
            self.backbone = {
                'convnext_tiny': convnext_tiny,
                'convnext_small': convnext_small,
                'convnext_base': convnext_base,
                'convnext_large': convnext_large,
                'convnext_xlarge': convnext_xlarge,
                'shufflenet_v2_x0_5': shufflenet_v2_x0_5,
                'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
                'shufflenet_v2_x1_5': shufflenet_v2_x1_5,
                'shufflenet_v2_x2_0': shufflenet_v2_x2_0,
                'mobilenet_v2': mobilenet_v2,
                'mobile_vit_xx_small': mobile_vit_xx_small,
                'mobile_vit_x_small': mobile_vit_x_small,
                'mobile_vit_small': mobile_vit_small,
                'RepVGG_A0': create_RepVGG_A0,
                'RepVGG_A1': create_RepVGG_A1,
                'RepVGG_A2': create_RepVGG_A2,
                'RepVGG_B0': create_RepVGG_B0,
                'RepVGG_B1': create_RepVGG_B1,
                'RepVGG_B1g2': create_RepVGG_B1g2,
                'RepVGG_B1g4': create_RepVGG_B1g4,
                'RepVGG_B2': create_RepVGG_B2,
                'RepVGG_B2g2': create_RepVGG_B2g2,
                'RepVGG_B2g4': create_RepVGG_B2g4,
                'RepVGG_B3': create_RepVGG_B3,
                'RepVGG_B3g2': create_RepVGG_B3g2,
                'RepVGG_B3g4': create_RepVGG_B3g4,
                'RepVGG_D2se': create_RepVGG_D2se,
                'mobilenet_v3_small': mobilenet_v3_small,
                'mobilenet_v3_large': mobilenet_v3_large,
                'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
                'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
                'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
                'swin_base_patch4_window12_384': swin_base_patch4_window12_384,
                'swin_base_patch4_window7_224_in22k': swin_base_patch4_window7_224_in22k,
                'swin_base_patch4_window12_384_in22k': swin_base_patch4_window12_384_in22k,
                'swin_large_patch4_window7_224_in22k': swin_large_patch4_window7_224_in22k,
                'swin_large_patch4_window12_384_in22k': swin_large_patch4_window12_384_in22k,
                'resnet18': resnet18,
                'resnet34': resnet34,
                'resnet50': resnet50,
                'resnet101': resnet101,
                'resnet152': resnet152,
                'densenet121': densenet121,
                'densenet161': densenet161,
                'densenet169': densenet169,
                'densenet201': densenet201,
            }[backbone](pretrained=pretrained, num_classes=num_classes)
            in_channels = {
                'convnext_tiny': [192, 384, 768],
                'convnext_small': [192, 384, 768],
                'convnext_base': [256, 512, 1024],
                'convnext_large': [384, 768, 1536],
                'convnext_xlarge': [512, 1024, 2048],
                'shufflenet_v2_x0_5': [48, 96, 192],
                'shufflenet_v2_x1_0': [116, 232, 464],
                'shufflenet_v2_x1_5': [176, 352, 704],
                'shufflenet_v2_x2_0': [244, 488, 976],
                'mobilenet_v2': [32, 96, 320],
                'mobile_vit_xx_small': [48, 64, 320],
                'mobile_vit_x_small': [64, 80, 384],
                'mobile_vit_small': [96, 128, 640],
                'RepVGG_A0': [96, 192, 1280],
                'RepVGG_A1': [128, 256, 1280],
                'RepVGG_A2': [192, 384, 1408],
                'RepVGG_B0': [128, 256, 1280],
                'RepVGG_B1': [256, 512, 2048],
                'RepVGG_B1g2': [256, 512, 2048],
                'RepVGG_B1g4': [256, 512, 2048],
                'RepVGG_B2': [320, 640, 2560],
                'RepVGG_B2g2': [320, 640, 2560],
                'RepVGG_B2g4': [320, 640, 2560],
                'RepVGG_B3': [384, 768, 2560],
                'RepVGG_B3g2': [384, 768, 2560],
                'RepVGG_B3g4': [384, 768, 2560],
                'RepVGG_D2se': [320, 640, 96],
                'mobilenet_v3_small': [24, 48, 96],
                'mobilenet_v3_large': [40, 112, 160],
                'swin_tiny_patch4_window7_224': [192, 384, 768],
                'swin_small_patch4_window7_224': [192, 384, 768],
                'swin_base_patch4_window7_224': [256, 512, 1024],
                'swin_base_patch4_window12_384': [256, 512, 1024],
                'swin_base_patch4_window7_224_in22k': [256, 512, 1024],
                'swin_base_patch4_window12_384_in22k': [256, 512, 1024],
                'swin_large_patch4_window7_224_in22k': [384, 768, 1536],
                'swin_large_patch4_window12_384_in22k': [384, 768, 1536],
                'resnet18': [128, 256, 512],
                'resnet34': [128, 256, 512],
                'resnet50': [512, 1024, 2048],
                'resnet101': [512, 1024, 2048],
                'resnet152': [512, 1024, 2048],
                'densenet121': [512, 1024, 1024],
                'densenet161': [768, 2112, 2208],
                'densenet169': [512, 1280, 1664],
                'densenet201': [512, 1792, 1920],
            }[backbone]
            feat1_c, feat2_c, feat3_c = in_channels
            self.conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
            self.conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            self.conv_1x1_feat3 = Conv(feat3_c, int(base_channels * 16 * deep_mul), 1, 1)

        if neck in {"FPN", "MobileFPN", "AttentionFPN"}:
            # ---------------------------------------------------#
            #   生成CSPdarknet53的主干模型
            #   获得三个有效特征层，他们的shape分别是：
            #   80,80,256
            #   40,40,512
            #   20,20,1024
            # ---------------------------------------------------#
            #self.backbone = Backbone(base_channels, base_depth, deep_mul, phi, pretrained)
            self.neck = {
                'FPN': FPN,
                'MobileFPN': MobileFPN,
                'AttentionFPN': AttentionFPN,
            }[neck](base_channels, base_depth, deep_mul, phi, pretrained)
        #else:
            # ---------------------------------------------------#
            #   如果输入不为cspdarknet，则调整通道数
            #   使其符合YoloV5的格式
            # ---------------------------------------------------#

        
        ch              = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape      = None
        self.nl         = len(ch)
        # self.stride     = torch.zeros(self.nl)
        self.stride     = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
        self.reg_max    = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no         = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.num_classes = num_classes
        
        c2, c3   = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()


    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self
    
    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)

        #print(feat1.shape)
        #print(feat2.shape)
        #print(feat3.shape)

        if self.backbone_name in {"swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224", "swin_base_patch4_window7_224", "swin_base_patch4_window12_384", "swin_base_patch4_window7_224_in22k", "swin_base_patch4_window12_384_in22k", "swin_large_patch4_window7_224_in22k", "swin_large_patch4_window12_384_in22k"}:
            channel_feature32 = feat3.size()[2]
            channel_feature16 = feat2.size()[2]
            channel_feature8 = feat1.size()[2]

            feature32x_sqrt = int(math.sqrt(feat3.size()[1]))
            feature16x_sqrt = int(math.sqrt(feat2.size()[1]))
            feature8x_sqrt = int(math.sqrt(feat1.size()[1]))

            feat3 = feat3.permute(0, 2, 1).contiguous().view(-1, channel_feature32, feature32x_sqrt,
                                                                       feature32x_sqrt)
            # print("after reshape feature32:", feature32x.size())
            feat2 = feat2.permute(0, 2, 1).contiguous().view(-1, channel_feature16, feature16x_sqrt,
                                                                       feature16x_sqrt)
            # print("after reshape feature16:", feature16x.size())
            feat1 = feat1.permute(0, 2, 1).contiguous().view(-1, channel_feature8, feature8x_sqrt, feature8x_sqrt)
            # print("after reshpae feature8:", feature8x.size())
        # print(feat1.shape)
        # print(feat2.shape)
        # print(feat3.shape)#
        if not self.backbone_name in {"cspdarknet", "DeformableCSPDarkNet", "MobileCSPDarkNet"}:
            feat1 = self.conv_1x1_feat1(feat1)
            feat2 = self.conv_1x1_feat2(feat2)
            feat3 = self.conv_1x1_feat3(feat3)

        #print(feat1.shape)
        #print(feat2.shape)
        #print(feat3.shape)#
        #------------------------加强特征提取网络------------------------# 
        P3, P4, P5 = self.neck(feat1, feat2, feat3)
        #------------------------加强特征提取网络------------------------#

        # P3 256, 80, 80
        # P4 512, 40, 40
        # P5 1024 * deep_mul, 20, 20
        shape = P3.shape  # BCHW
        
        # P3 256, 80, 80 => num_classes + self.reg_max * 4, 80, 80
        # P4 512, 40, 40 => num_classes + self.reg_max * 4, 40, 40
        # P5 1024 * deep_mul, 20, 20 => num_classes + self.reg_max * 4, 20, 20
        x = [P3, P4, P5]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        
        # num_classes + self.reg_max * 4 , 8400 =>  cls num_classes, 8400; 
        #                                           box self.reg_max * 4, 8400
        box, cls        = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.num_classes), 1)
        # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]
        dbox            = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)