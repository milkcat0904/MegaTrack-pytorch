import os
import torch
import torch.nn as nn
from torchsummary import summary

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class KeyNet(nn.Modules):
    def __init__(self, hyper):
        super(KeyNet, self).__init__()

        self.image_network = self.make_backbone_image()
        self.keypoints_network = self.make_backbone_keypoints()
        self.fused_network = self.make_backbone_fused()
        self.new_network = self.newhand_regression()
        self.heatmap_network = self.heatmap_regression()

        if hyper['model']['resume'] == False:
            print ('Init para...')
            self.init_weights()

    def newhand_regression(self):
        out = nn.Sequential(
            nn.AvgPool2d(kernel_size = 6, stride = 6),

            nn.Conv2d(in_channels = 160, out_channels = 378, kernel_size = 1),
            nn.ReLU6(inplace = True),
            nn.Conv2d(in_channels = 378, out_channels = 128, kernel_size = 1),
            nn.ReLU6(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 63, kernel_size = 1)
        )
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            #
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_backbone_image(self):

        input_channel = 32
        interverted_residual_setting = [
            # t, c, n, s
            [1, 32, 1, 1],
            [1, 32, 1, 2],
            [1, 32, 1, 1],
            [1, 64, 1, 2],
            [1, 64, 2, 1],
        ]

        out = [
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

    def make_backbone_keypoints(self):
        out = nn.Sequential(
            nn.Linear(63, 4608),
            nn.ReLU6(inplace = True)
        )
        return out

    def make_backbone_fused(self, input_channel=96):
        interverted_residual_setting = [
            # t, c, n, s
            [1, 64, 2, 1],
            [1, 64, 3, 1],
            [1, 96, 1, 1],
            [1, 96, 2, 1],
            [1, 128, 1, 2],
            [1, 128, 2, 1],
            [1, 160, 1, 1],
        ]

        out = []
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    out.append(InvertedResidual(input_channel, output_channel, s, expand_ratio = t))
                else:
                    out.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio = t))
                input_channel = output_channel

        return nn.Sequential(*out)

    def heatmap_regression(self):
        out = nn.Sequential(
            nn.Conv2d(in_channels = 160, out_channels = 63, kernel_size = 3, padding = 2),
            nn.BatchNorm2d(63),
            nn.ReLU6(inplace = True),

            nn.ConvTranspose2d(in_channels = 63, out_channels = 42, kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 42, out_channels = 21, kernel_size = 3, padding = 2),
            nn.BatchNorm2d(21),
            nn.ReLU6(inplace = True)
        )
        return out

    def forward(self, x, addon=torch.ones(2, 63)):
        x = self.image_network(x)
        x_addon = self.keypoints_network(addon) # b, 4608
        x_addon = x_addon.view(-1, 32, 12, 12)

        x = torch.cat((x, x_addon), dim = 1) # b, 96, 12, 12
        x = self.fused_network(x)

        x_new = self.new_network(x).squeeze(dim=2).squeeze(dim=2) # b 63 1 1 -> b 63
        x_hp = self.heatmap_network(x)

        return x_new, x_hp