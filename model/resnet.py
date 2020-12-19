import torch
from torch import nn


def conv3x3(in_ch, out_ch, stride=1, dilation=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, dilation=dilation)


def conv1x1(in_ch, out_ch, stride=1, dilation=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = norm_layer(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch*self.expansion)
        self.bn2 = norm_layer(out_ch*self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BottleBlock(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1, norm_layer=None):
        super(BottleBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(in_ch, out_ch, stride)
        self.bn1 = norm_layer(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = norm_layer(out_ch)

        self.conv3 = conv1x1(out_ch, out_ch*self.expansion)
        self.bn3 = norm_layer(out_ch*self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.in_ch = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(3, self.in_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[1], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out_ch, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample =None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_ch != out_ch * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_ch, out_ch*block.expansion, stride), norm_layer(out_ch*block.expansion))
        layers = []
        layers.append(block(self.in_ch, out_ch*block.expansion, downsample=downsample, norm_layer=norm_layer))
        self.in_ch = out_ch * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_ch, out_ch, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = self.fc(out)
        return out


def resnet18(layers=[2, 2, 2, 2], num_classes=2):
    return ResNet(BasicBlock, layers, num_classes)


def resnet34(layers=[3, 4, 6, 3], num_classes=2):
    return ResNet(BasicBlock, layers, num_classes)


def resnet50(layers=[3, 4, 6, 3], num_classes=2):
    return ResNet(BottleBlock, layers, num_classes)


if __name__ == "__main__":
    net = resnet18(BasicBlock)
    print(net)




