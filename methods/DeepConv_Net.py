import torch.nn as nn


class DeepConvNet(nn.Module):
    """PyTorch implementation of the adapted DeepConvNet architecture."""

    def __init__(self, config):
        super().__init__()
        assert (hasattr(config, 'fs') and
                hasattr(config, 'frame') and
                hasattr(config, 'CH') and
                hasattr(config, 'dropoutRate'))

        self.features = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 25, kernel_size=(config.CH, 1)),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(config.dropoutRate),

            nn.Conv2d(25, 50, kernel_size=(1, 5)),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(config.dropoutRate),

            nn.Conv2d(50, 100, kernel_size=(1, 5)),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(config.dropoutRate),

            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(config.dropoutRate),
            nn.Flatten(),
        )
        self.classifier = nn.LazyLinear(2)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def net(config):
    return DeepConvNet(config)
