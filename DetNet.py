import torch
import torch.nn as nn
from modules.irb import InvertedResidual

class DetNet(nn.Modules):
    def __init__(self):
        super(DetNet, self).__init__()

        self.backbone = self.make_backbone()
        self.center_network = self.make_network()
        self.radius_network = self.make_network()
        self.cls_network = self.make_cls_network()

    def make_backbone(self):
        input_channel = 32
        interverted_residual_setting = [
            # t, c, n, s
            [1, 32, 1, 2],
            [6, 32, 1, 2],
            [6, 32, 1, 1],
            [6, 64, 1, 2],
            [6, 64, 2, 1],
            [6, 64, 1, 2],
            [6, 64, 3, 1],
            [6, 96, 1, 1],
            [6, 96, 2, 1],
            [6, 128, 1, 2],
            [6, 128, 2, 1],
            [6, 160, 1, 1],
        ]

        out = [
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(1, input_channel, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace = True)
        ]

        for t, c, n, s in interverted_residual_setting:

            output_channel = c
            for i in range(n):
                if i == 0:
                    out.append(InvertedResidual(input_channel, output_channel, s, expand_ratio = t))
                else:
                    out.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio = t))
                input_channel = output_channel
        return nn.Sequential(*out)

    def make_network(self):
        out = nn.Sequential(
            nn.Conv2d(160, 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU6(inplace = True),
            nn.AdaptiveAvgPool2d(4),
        )
        return out

    def make_cls_network(self):
        out = nn.Sequential(
            nn.Conv2d(160, 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU6(inplace = True),
            nn.AdaptiveAvgPool2d(4),
            nn.Sigmoid()
        )
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        center = self.center_network(x).view(-1, 2, 2)
        radius = self.radius_network(x)
        cls = self.cls_network(x)

        return center, radius, cls

