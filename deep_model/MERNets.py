import torch
import torch.nn as nn
from torch.nn import functional as F

################ Architectures used in outside ################
__all__ = ['STSTNet', 'DualInceptionNet', 'RCNFNet']

################ Blocks used inside ################
class ConvBlock(nn.Module):
    """convolutional layer blocks for sequtial convolution operations"""
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)

class RclBlock(nn.Module):
    """recurrent convolutional blocks"""
    def __init__(self, inplanes, planes):
        super(RclBlock, self).__init__()
        self.ffconv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.rrconv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout()
        )

    def forward(self, x):
        y = self.ffconv(x)
        y = self.rrconv(x + y)
        y = self.rrconv(x + y)
        out = self.downsample (y)
        return out

class DenseBlock(nn.Module):
    """densely connected convolutional blocks"""
    def __init__(self, inplanes, planes):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout()
        )

    def forward(self, x):
        y = self.conv1(x)
        z = self.conv2(x + y)
        # out = self.conv2(x + y + z)
        e = self.conv2(x + y + z)
        out = self.conv2(x + y + z + e)
        out = self.downsample (out)
        return out

class SpatialAttentionBlock(nn.Module):
    """linear attention block for any layers"""
    def __init__(self, normalize_attn=True):
        super(SpatialAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn

    def forward(self, l, w, classes):
        output_cam = []
        for idx in range(0,classes):
            weights = w[idx,:].reshape((l.shape[1], l.shape[2], l.shape[3]))
            cam = weights * l
            cam = cam.mean(dim=1,keepdim=True)
            cam = cam - torch.min(torch.min(cam,3,True)[0],2,True)[0]
            cam = cam / torch.max(torch.max(cam,3,True)[0],2,True)[0]
            output_cam.append(cam)
        output = torch.cat(output_cam, dim=1)
        output = output.mean(dim=1,keepdim=True)
        return output

class EmbeddingBlock(nn.Module):
    """densely connected convolutional blocks for embedding"""
    def __init__(self, inplanes, planes):
        super(EmbeddingBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.attenmap = SpatialAttentionBlock(normalize_attn=True)
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout()
        )

    def forward(self, x, w, pool_size, classes):
        y = self.conv1(x)
        y1 = self.attenmap(F.adaptive_avg_pool2d(x, (pool_size, pool_size)), w, classes)
        y = torch.mul(F.interpolate(y1, (y.shape[2], y.shape[3])), y)
        z = self.conv2(x+y)
        e = self.conv2(x + y + z)
        out = self.conv2(x + y + z + e)
        out = self.downsample (out)
        return out

def MakeLayer(block, planes, blocks):
    layers = []
    for _ in range(0, blocks):
        layers.append(block(planes, planes))
    return nn.Sequential(*layers)

################ Architecture definiations ################
class STSTNet(nn.Module):
    """ShallowNet network published in
        "Shallow Triple Stream Three-dimensional CNN (STSTNet) for Micro-expression Recognition"
    """
    def __init__(self, pretrained=False, num_classes=3, init_weights=True):
        super(STSTNet, self).__init__()
        self.steam1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.steam2 = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(5),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.steam3 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Linear(400, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.steam1 (x)
        x2 = self.steam2(x)
        x3 = self.steam3(x)
        x = torch.cat((x1,x2,x3),1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # return torch.squeeze(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class DualInceptionNet(nn.Module):
    """DualInceptionNet network published in
        "Dual-Inception Network for Cross-Database Micro-Expression Recognition"
    """
    def __init__(self, num_input, num_classes):
        super(DualInceptionNet, self).__init__()
        # first stream
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(num_input, 6, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(num_input, 6, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(num_input, 6, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        self.conv1_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_input, 6, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        self.conv1_8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        # second stream
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(num_input, 6, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(num_input, 6, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(num_input, 6, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        self.conv2_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_input, 6, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv2_5 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv2_6 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2_7 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        self.conv2_8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.norm = nn.ReLU(inplace=True)
        self.linear = nn.Linear(3200, 1024)
        self.classifier = nn.Linear(1024, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x1 = torch.cat((self.conv1_1(x),self.conv1_2(x),self.conv1_3(x),self.conv1_4(x)),1)
        x1 = self.maxpool(x1)
        x1 = torch.cat((self.conv1_5(x1), self.conv1_6(x1), self.conv1_7(x1), self.conv1_8(x1)), 1)
        x1 = self.maxpool(x1)
        x2 = torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), 1)
        x2 = self.maxpool(x2)
        x2 = torch.cat((self.conv2_5(x2), self.conv2_6(x2), self.conv1_7(x2), self.conv2_8(x2)), 1)
        x2 = self.maxpool(x2)
        x = torch.cat((torch.flatten(x1,1),torch.flatten(x2,1)),1)
        x = self.linear(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class RCNFNet(nn.Module):
    """RCNFNet network published in
        "Revealing the Invisible With Model and Data Shrinking for Composite-Database Micro-Expression Recognition"
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5):
        super(RCNFNet, self).__init__()
        self.classes = num_classes
        self.poolsize = pool_size
        num_channels = int(featuremaps/2)
        self.stream1 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            nn.Dropout(),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(num_input, int(num_channels/2), kernel_size=3, stride=3, padding=2, dilation=2),  # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(num_channels/2)),
            nn.Dropout(),
        )
        self.stream3 = nn.Sequential(
            nn.Conv2d(num_input, int(num_channels/2), kernel_size=3, stride=3, padding=3, dilation=3),  # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(num_channels/2)),
            nn.Dropout(),
        )
        self.ebl = EmbeddingBlock(featuremaps, featuremaps)
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        # x = torch.cat((x1, x2), 1)
        x3 = self.stream2(x)
        x = torch.cat((x1,x2,x3),1)
        x = self.ebl(x, self.classifier.weight, self.poolsize, self.classes)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x