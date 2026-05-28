import copy

import torch
import torch.nn as nn

from methods.foundation_utils import load_matching_state_dict, maybe_resample


def add_cli_args(parser):
    group = parser.add_argument_group('BENDR method arguments')
    group.add_argument('--bendr-pretrained', default='',
                       help='Optional BENDR encoder checkpoint path.')
    group.add_argument('--bendr-target-fs', type=int, default=256,
                       help='Sampling frequency used inside the BENDR adapter.')
    group.add_argument('--bendr-hidden', type=int, default=512,
                       help='BENDR convolutional encoder hidden size.')


def apply_cli_args(config, args):
    config.bendr_pretrained = args.bendr_pretrained
    config.bendr_target_fs = args.bendr_target_fs
    config.bendr_hidden = args.bendr_hidden


class ConvEncoderBENDR(nn.Module):
    def __init__(self, in_features, encoder_h=512, enc_width=(3, 2, 2, 2, 2, 2),
                 dropout=0.0, enc_downsample=(3, 2, 2, 2, 2, 2)):
        super().__init__()
        enc_width = [w if w % 2 else w + 1 for w in enc_width]
        self.encoder = nn.Sequential()
        for idx, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module(f'Encoder_{idx}', nn.Sequential(
                nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                nn.Dropout1d(dropout),
                nn.GroupNorm(max(1, encoder_h // 2), encoder_h),
                nn.GELU(),
            ))
            in_features = encoder_h

    def forward(self, x):
        return self.encoder(x)


class _NoNorm(nn.Module):
    def forward(self, x):
        return x


class BENDRContextualizer(nn.Module):
    def __init__(self, in_features, hidden_feedforward=3076, heads=8, layers=4, dropout=0.15):
        super().__init__()
        transformer_dim = in_features * 3
        encoder = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=heads,
            dim_feedforward=hidden_feedforward,
            dropout=dropout,
            activation='gelu',
        )
        encoder.norm1 = _NoNorm()
        encoder.norm2 = _NoNorm()
        self.layers = nn.ModuleList([copy.deepcopy(encoder) for _ in range(layers)])
        self.relative_position = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(in_features, in_features, 25, padding=12, groups=16), dim=2),
            nn.GELU(),
        )
        self.input_conditioning = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, transformer_dim),
        )
        self.output_layer = nn.Conv1d(transformer_dim, in_features, 1)

    def forward(self, x):
        x = x + self.relative_position(x)
        x = self.input_conditioning(x.transpose(1, 2)).transpose(0, 1)
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x.permute(1, 2, 0))


class BendrClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = getattr(config, 'CH', 2)
        hidden = getattr(config, 'bendr_hidden', 512)
        self.input_fs = getattr(config, 'fs', None)
        self.target_fs = getattr(config, 'bendr_target_fs', 256)
        self.encoder = ConvEncoderBENDR(channels, encoder_h=hidden)
        self.contextualizer = BENDRContextualizer(hidden, layers=4)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, getattr(config, 'num_classes', 2)),
        )
        load_matching_state_dict(self.encoder, getattr(config, 'bendr_pretrained', ''))

    def forward(self, x):
        x = maybe_resample(x, self.input_fs, self.target_fs)
        x = self.encoder(x)
        x = self.contextualizer(x)
        x = x[:, :, -1]
        return self.classifier(x)


def net(config):
    return BendrClassifier(config)
