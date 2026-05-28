import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.foundation_utils import load_matching_state_dict, maybe_resample


def add_cli_args(parser):
    group = parser.add_argument_group('CBraMod method arguments')
    group.add_argument('--cbramod-pretrained', default='',
                       help='Optional CBraMod checkpoint path.')
    group.add_argument('--cbramod-target-fs', type=int, default=200,
                       help='Sampling frequency used inside the CBraMod adapter.')
    group.add_argument('--cbramod-patch-size', type=int, default=200,
                       help='CBraMod temporal patch size.')


def apply_cli_args(config, args):
    config.cbramod_pretrained = args.cbramod_pretrained
    config.cbramod_target_fs = args.cbramod_target_fs
    config.cbramod_patch_size = args.cbramod_patch_size


def _get_clones(module, count):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


class CrissCrossEncoderLayer(nn.Module):
    def __init__(self, d_model=200, nhead=8, dim_feedforward=800, dropout=0.1):
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, batch_first=True)
        self.self_attn_t = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        x = src
        y = self.norm1(x)
        batch, channels, patches, patch_size = y.shape
        ys = y[:, :, :, :patch_size // 2]
        yt = y[:, :, :, patch_size // 2:]
        ys = ys.transpose(1, 2).reshape(batch * patches, channels, patch_size // 2)
        yt = yt.reshape(batch * channels, patches, patch_size // 2)
        ys = self.self_attn_s(ys, ys, ys, need_weights=False)[0].reshape(batch, patches, channels, patch_size // 2).transpose(1, 2)
        yt = self.self_attn_t(yt, yt, yt, need_weights=False)[0].reshape(batch, channels, patches, patch_size // 2)
        x = x + self.dropout(torch.cat((ys, yt), dim=3))
        x = x + self.dropout(self.linear2(F.gelu(self.linear1(self.norm2(x)))))
        return x


class CBraModBackbone(nn.Module):
    def __init__(self, in_dim=200, d_model=200, n_layer=4, nhead=8):
        super().__init__()
        self.in_dim = in_dim
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Linear(in_dim // 2 + 1, d_model)
        layer = CrissCrossEncoderLayer(d_model=d_model, nhead=nhead)
        self.layers = _get_clones(layer, n_layer)

    def forward(self, x):
        batch, channels, patches, patch_size = x.shape
        conv_in = x.reshape(batch, 1, channels * patches, patch_size)
        patch_emb = self.proj_in(conv_in).permute(0, 2, 1, 3).reshape(batch, channels, patches, self.in_dim)
        spectral = torch.abs(torch.fft.rfft(x.reshape(batch * channels * patches, patch_size), dim=-1, norm='forward'))
        spectral = self.spectral_proj(spectral).reshape(batch, channels, patches, self.in_dim)
        x = patch_emb + spectral
        for layer in self.layers:
            x = layer(x)
        return x


class CBraModClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_fs = getattr(config, 'fs', None)
        self.target_fs = getattr(config, 'cbramod_target_fs', 200)
        self.patch_size = getattr(config, 'cbramod_patch_size', 200)
        channels = getattr(config, 'CH', 2)
        self.backbone = CBraModBackbone(in_dim=self.patch_size, d_model=self.patch_size, n_layer=4, nhead=8)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels * self.patch_size, self.patch_size),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(self.patch_size, 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(200, getattr(config, 'num_classes', 2)),
        )
        load_matching_state_dict(self.backbone, getattr(config, 'cbramod_pretrained', ''))

    def forward(self, x):
        x = maybe_resample(x, self.input_fs, self.target_fs)
        if x.shape[-1] < self.patch_size:
            x = F.pad(x, (0, self.patch_size - x.shape[-1]))
        x_windows = x.unfold(dimension=2, size=self.patch_size, step=self.patch_size)
        feats = self.backbone(x_windows).mean(dim=2)
        return self.feed_forward(feats.reshape(feats.shape[0], -1))


def net(config):
    return CBraModClassifier(config)
