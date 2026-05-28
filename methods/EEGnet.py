import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class EEGNet(nn.Module):
    def __init__(self, config, dropoutRate=0.5, kernLength=64, F1=8,
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super().__init__()
        kernLength = int(config.fs / 2)

        if dropoutType == 'SpatialDropout2D':
            dropout_layer = nn.Dropout2d
        elif dropoutType == 'Dropout':
            dropout_layer = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')

        self.features = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(config.CH, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            dropout_layer(dropoutRate),
            SeparableConv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            dropout_layer(dropoutRate),
            nn.Flatten(),
        )
        self.classifier = nn.LazyLinear(2)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def net(config, dropoutRate=0.5, kernLength=64, F1=8,
        D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    return EEGNet(config, dropoutRate, kernLength, F1, D, F2, norm_rate, dropoutType)
