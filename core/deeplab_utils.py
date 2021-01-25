# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_fn=None):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_fn(planes)
        self.relu = nn.ReLU(inplace=True)

        self.initialize([self.atrous_conv, self.bn])

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, output_stride, norm_fn):
        super().__init__()

        inplanes = 2048

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        
        self.aspp1 = ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], norm_fn=norm_fn)
        self.aspp2 = ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], norm_fn=norm_fn)
        self.aspp3 = ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], norm_fn=norm_fn)
        self.aspp4 = ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], norm_fn=norm_fn)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
            norm_fn(256),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = norm_fn(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
        self.initialize([self.conv1, self.bn1] + list(self.global_avg_pool.modules()))
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_inplanes, norm_fn):
        super().__init__()
        
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = norm_fn(48)
        self.relu = nn.ReLU(inplace=True)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_fn(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_fn(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

        self.initialize([self.conv1, self.bn1] + list(self.classifier.modules()))

    def forward(self, x, x_low_level):
        x_low_level = self.conv1(x_low_level)
        x_low_level = self.bn1(x_low_level)
        x_low_level = self.relu(x_low_level)

        x = F.interpolate(x, size=x_low_level.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, x_low_level), dim=1)
        x = self.classifier(x)

        return x

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()