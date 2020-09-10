import torch
import torch.nn as nn


class ShallowNet(nn.Module):

    def __init__(self, pretrained=False, num_classes=3, init_weights=True):
        super(ShallowNet, self).__init__()
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


class WideNet(nn.Module):

    def __init__(self, pretrained=False, num_classes=3, init_weights=True):
        super(WideNet, self).__init__()
        self.steam1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.steam2 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.steam3 = nn.Sequential(
            nn.Conv2d(5, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.steam4 = nn.Sequential(
            nn.Conv2d(7, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Linear(400, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.steam1(x)
        x2 = self.steam2(x)
        x3 = self.steam3(x)
        x4 = self.steam4(x)
        x = torch.cat((x1,x2,x3,x4),1)
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