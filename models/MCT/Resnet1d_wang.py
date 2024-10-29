## https://github.com/hhz936/CIMIL/blob/6141c8d50aec448c4c7f8a027efeea43e81c5f9a/resnet1d_wang.py#L89
## Resnet1d_wang.py 오픈 소스에 VIT에 넣을 수 있게 토큰으로 출력하게 만듬

import torch.nn as nn
import torch

def conv(in_planes, out_planes, stride=1, kernel_size=3):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size - 1) // 2, bias=False)


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=[3, 3], downsample=None):
        super(BasicBlock1d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size // 2 + 1]

        self.conv1 = conv(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes, kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet1d_Wang(nn.Module):
    def __init__(self, block, layers, kernel_size=5, input_channels=1, token_dim=256): 
        super(ResNet1d_Wang, self).__init__()
        self.inplanes = 64
        self.conv = nn.Conv1d(input_channels, 64, kernel_size=8, stride=2, padding=7 // 2, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.block1 = self._make_layer(block, 64, layers[0], kernel_size=kernel_size)
        self.block2 = self._make_layer(block, 128, layers[1], stride=2, kernel_size=kernel_size)
        self.block3 = self._make_layer(block, token_dim, layers[2], stride=2, kernel_size=kernel_size)
        # self.Avgpool은 제거하여 시퀀스 형태 그대로 유지

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=3):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, kernel_size, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # 최종 출력 형태는 (batch_size, token_dim, sequence_length)로 토큰화된 특징 벡터를 반환
        return x

