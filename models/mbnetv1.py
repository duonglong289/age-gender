# coding: utf-8

from __future__ import division

import math
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MobileNetV1', 'mobilenetv1']
# __all__ = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']


class DepthWiseBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, prelu=False):
        super(DepthWiseBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes,
                                 bias=False)
        self.bn_dw = nn.BatchNorm2d(inplanes)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)

        out = self.conv_sep(out)
        out = self.bn_sep(out)
        out = self.relu(out)

        return out


class MobileNetV1(nn.Module):
    def __init__(self, widen_factor=1.0, num_age_classes=100, num_gender_classes=2, prelu=False, input_channel=3):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNetV1, self).__init__()

        block = DepthWiseBlock
        self.conv1 = nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(int(32 * widen_factor))
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, prelu=prelu)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, prelu=prelu)

        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, prelu=prelu)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, prelu=prelu)

        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, prelu=prelu)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, prelu=prelu)

        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, prelu=prelu)

        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, prelu=prelu)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.last_channel = 1024*widen_factor
        # building classifier for age and gender
        self.age_cls = False
        if num_age_classes is not None:
            self.age_cls = True
            self.classifier_age = nn.Sequential(
                nn.Linear(self.last_channel, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_age_classes))

        if num_gender_classes is not None:
            self.gender_cls = True
            self.classifier_gender = nn.Sequential(
                nn.Linear(self.last_channel, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_gender_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dw2_1(x)
        x = self.dw2_2(x)
        x = self.dw3_1(x)
        x = self.dw3_2(x)
        x = self.dw4_1(x)
        x = self.dw4_2(x)
        x = self.dw5_1(x)
        x = self.dw5_2(x)
        x = self.dw5_3(x)
        x = self.dw5_4(x)
        x = self.dw5_5(x)
        x = self.dw5_6(x)
        x = self.dw6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Classifier
        age_score = self.classifier_age(x)
        age_prob = F.log_softmax(age_score, dim=1)
        
        gender_score = self.classifier_gender(x)
        gender_prob = F.log_softmax(gender_score, dim=1)

        return age_score, age_prob, gender_prob


def mobilenet_v1(**kwargs):
    """
    Construct MobileNet.
    widen_factor=1.0  for mobilenet_1
    widen_factor=0.75 for mobilenet_075
    widen_factor=0.5  for mobilenet_05
    widen_factor=0.25 for mobilenet_025
    """
    # widen_factor = 1.0, num_classes = 1000
    # model = MobileNet(widen_factor=widen_factor, num_classes=num_classes)
    # return model

    model = MobileNetV1(
        widen_factor=kwargs.get('widen_factor', 1.0),
        num_age_classes=kwargs.get('num_age_classes', 100),
        num_gender_classes=kwargs.get('num_gender_classes', 2)
    )
    return model


def mobilenet_2(num_age_classes=100, num_gender_classes=2, input_channel=3):
    model = MobileNetV1(widen_factor=2.0, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_1(num_age_classes=100, num_gender_classes=2, input_channel=3):
    model = MobileNetV1(widen_factor=1.0, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_075(num_age_classes=100, num_gender_classes=2, input_channel=3):
    model = MobileNetV1(widen_factor=0.75, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_05(num_age_classes=100, num_gender_classes=2, input_channel=3):
    model = MobileNetV1(widen_factor=0.5, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_025(num_age_classes=100, num_gender_classes=2, input_channel=3):
    model = MobileNetV1(widen_factor=0.25, num_classes=num_classes, input_channel=input_channel)
    return model
