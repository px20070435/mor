import math

import torch
import torch.nn as nn

from methods.foundation_utils import load_matching_state_dict, maybe_resample


def add_cli_args(parser):
    group = parser.add_argument_group('EEGPT method arguments')
    group.add_argument('--eegpt-pretrained', default='',
                       help='Optional EEGPT checkpoint path.')
    group.add_argument('--eegpt-target-fs', type=int, default=256,
                       help='Sampling frequency used inside the EEGPT adapter.')
    group.add_argument('--eegpt-emb-size', type=int, default=512,
                       help='EEGPT embedding size.')
    group.add_argument('--eegpt-depth', type=int, default=8,
                       help='Number of EEGPT transformer layers.')
    group.add_argument('--eegpt-heads', type=int, default=8,
                       help='Number of EEGPT attention heads.')
    group.add_argument('--eegpt-patch-size', type=int, default=64,
                       help='EEGPT temporal patch size.')
    group.add_argument('--eegpt-channel-names', default='BTEleftSD,CROSStopSD',
                       help='Comma-separated channel names used for documentation and checkpoint compatibility.')


def apply_cli_args(config, args):
    config.eegpt_pretrained = args.eegpt_pretrained
    config.eegpt_target_fs = args.eegpt_target_fs
    config.eegpt_emb_size = args.eegpt_emb_size
    config.eegpt_depth = args.eegpt_depth
    config.eegpt_heads = args.eegpt_heads
    config.eegpt_patch_size = args.eegpt_patch_size
    config.eegpt_channel_names = [x.strip() for x in args.eegpt_channel_names.split(',') if x.strip()]


class SinusoidalTemporalEncoding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:, :length]


class EEGPTPatchEncoder(nn.Module):
    def __init__(self, channels, patch_size=64, emb_size=512, depth=8, heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(1, emb_size, kernel_size=(1, patch_size), stride=(1, patch_size))
        self.channel_embed = nn.Embedding(max(channels, 1), emb_size)
        self.time_embed = SinusoidalTemporalEncoding(emb_size)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=heads,
            dim_feedforward=4 * emb_size,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        if x.shape[-1] < self.patch_size:
            x = torch.nn.functional.pad(x, (0, self.patch_size - x.shape[-1]))
        x = self.patch_embed(x.unsqueeze(1))
        batch, emb, channels, patches = x.shape
        x = x.permute(0, 3, 2, 1)
        ch_idx = torch.arange(channels, device=x.device)
        x = x + self.channel_embed(ch_idx).view(1, 1, channels, emb)
        x = x + self.time_embed(patches).view(1, patches, 1, emb)
        x = x.reshape(batch, patches * channels, emb)
        return self.norm(self.blocks(x)).mean(dim=1)


class EEGPTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_fs = getattr(config, 'fs', None)
        self.target_fs = getattr(config, 'eegpt_target_fs', 256)
        channels = getattr(config, 'CH', 2)
        emb_size = getattr(config, 'eegpt_emb_size', 512)
        self.chan_conv = nn.Conv1d(channels, channels, 1)
        self.encoder = EEGPTPatchEncoder(
            channels=channels,
            patch_size=getattr(config, 'eegpt_patch_size', 64),
            emb_size=emb_size,
            depth=getattr(config, 'eegpt_depth', 8),
            heads=getattr(config, 'eegpt_heads', 8),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(emb_size, 16),
            nn.ELU(),
            nn.Linear(16, getattr(config, 'num_classes', 2)),
        )
        load_matching_state_dict(self, getattr(config, 'eegpt_pretrained', ''), keys=('state_dict', 'model'))

    def forward(self, x):
        x = maybe_resample(x, self.input_fs, self.target_fs)
        x = self.chan_conv(x)
        return self.head(self.encoder(x))


def net(config):
    return EEGPTClassifier(config)
