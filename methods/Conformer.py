import torch
import torch.nn as nn
import torch.nn.functional as F


def add_cli_args(parser):
    group = parser.add_argument_group('Conformer method arguments')
    group.add_argument('--conformer-emb-size', type=int, default=40,
                       help='Conformer embedding size.')
    group.add_argument('--conformer-depth', type=int, default=6,
                       help='Number of transformer blocks.')
    group.add_argument('--conformer-heads', type=int, default=10,
                       help='Number of attention heads.')


def apply_cli_args(config, args):
    config.conformer_emb_size = args.conformer_emb_size
    config.conformer_depth = args.conformer_depth
    config.conformer_heads = args.conformer_heads


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, num_channel=2):
        super().__init__()
        self.shallow_net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (num_channel, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 37), (1, 7)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.shallow_net(x.unsqueeze(1))
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, seq_len, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(seq_len * emb_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.fc(x.reshape(x.size(0), -1))


class Conformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = getattr(config, 'CH', 2)
        data_length = int(getattr(config, 'fs', 250) * getattr(config, 'frame', 2))
        emb_size = getattr(config, 'conformer_emb_size', 40)
        depth = getattr(config, 'conformer_depth', 6)
        heads = getattr(config, 'conformer_heads', 10)
        self.patch = PatchEmbedding(emb_size, channels)

        with torch.no_grad():
            seq_len = self.patch(torch.zeros(1, channels, data_length)).shape[1]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=heads,
            dim_feedforward=4 * emb_size,
            dropout=0.5,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.class_head = ClassificationHead(emb_size, seq_len, getattr(config, 'num_classes', 2))

    def forward(self, x):
        x = self.patch(x)
        x = self.transformer(x)
        return self.class_head(x)


def net(config):
    return Conformer(config)
