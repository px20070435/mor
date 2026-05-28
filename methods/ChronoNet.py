import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        out_len = (x.size(-1) + self.stride - 1) // self.stride
        pad_needed = max((out_len - 1) * self.stride + self.kernel_size - x.size(-1), 0)
        left = pad_needed // 2
        right = pad_needed - left
        return self.conv(F.pad(x, (left, right)))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, filters, stride, use_batchnorm=True, dropout=0.6):
        super().__init__()
        self.branches = nn.ModuleList([
            SamePadConv1d(in_channels, filters, kernel_size=2, stride=stride),
            SamePadConv1d(in_channels, filters, kernel_size=4, stride=stride),
            SamePadConv1d(in_channels, filters, kernel_size=8, stride=stride),
        ])
        out_channels = filters * len(self.branches)
        self.batchnorm = nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x):
        x = torch.cat([F.relu(branch(x)) for branch in self.branches], dim=1)
        x = self.batchnorm(x)
        return self.dropout(x)


class ChronoNet(nn.Module):
    ## Based on source: https://github.com/aguscerdo/EE239AS-Project
    def __init__(self, config, inception=True, res=True, strided=True, batchnorm=True):
        super().__init__()
        assert (hasattr(config, 'fs') and
                hasattr(config, 'frame') and
                hasattr(config, 'CH') and
                hasattr(config, 'dropoutRate'))

        self.res = res
        state_size = 32
        filters = 32
        stride = 2 if strided else 1
        cnn_drop = 0.6

        if inception:
            self.cnn = nn.Sequential(
                InceptionBlock(config.CH, filters, stride, batchnorm, cnn_drop),
                InceptionBlock(filters * 3, filters, stride, batchnorm, cnn_drop),
                InceptionBlock(filters * 3, filters, stride, batchnorm, cnn_drop),
            )
            rnn_input = filters * 3
        else:
            self.cnn = nn.Sequential(
                SamePadConv1d(config.CH, filters, kernel_size=4, stride=stride),
                nn.ReLU(),
                nn.BatchNorm1d(filters) if batchnorm else nn.Identity(),
                nn.Dropout1d(cnn_drop),
                SamePadConv1d(filters, filters, kernel_size=4, stride=stride),
                nn.ReLU(),
                nn.BatchNorm1d(filters) if batchnorm else nn.Identity(),
                nn.Dropout1d(cnn_drop),
                SamePadConv1d(filters, filters, kernel_size=4, stride=stride),
                nn.ReLU(),
                nn.BatchNorm1d(filters) if batchnorm else nn.Identity(),
                nn.Dropout1d(cnn_drop),
            )
            rnn_input = filters

        self.gru1 = nn.GRU(rnn_input, state_size, batch_first=True)
        if res:
            self.gru2 = nn.GRU(state_size, state_size, batch_first=True)
            self.gru3 = nn.GRU(state_size * 2, state_size, batch_first=True)
            self.gru4 = nn.GRU(state_size * 3, state_size, batch_first=True)
        else:
            self.gru2 = nn.GRU(state_size, state_size, batch_first=True)
            self.gru3 = nn.GRU(state_size, state_size, batch_first=True)
            self.gru4 = nn.GRU(state_size, state_size, batch_first=True)
        self.classifier = nn.Linear(state_size, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2)
        g1, _ = self.gru1(x)
        g2, _ = self.gru2(g1)
        if self.res:
            g3, _ = self.gru3(torch.cat([g1, g2], dim=-1))
            _, h = self.gru4(torch.cat([g1, g2, g3], dim=-1))
        else:
            g3, _ = self.gru3(g2)
            _, h = self.gru4(g3)
        return self.classifier(h[-1])


def net(config, inception=True, res=True, strided=True, maxpool=False, avgpool=False, batchnorm=True):
    return ChronoNet(config, inception=inception, res=res, strided=strided, batchnorm=batchnorm)
