import math

import torch
import torch.nn as nn

from methods.foundation_utils import load_matching_state_dict, maybe_resample


def add_cli_args(parser):
    group = parser.add_argument_group('BIOT method arguments')
    group.add_argument('--biot-pretrained', default='',
                       help='Optional BIOT checkpoint path.')
    group.add_argument('--biot-target-fs', type=int, default=200,
                       help='Sampling frequency used inside the BIOT adapter.')
    group.add_argument('--biot-emb-size', type=int, default=256,
                       help='BIOT embedding size.')
    group.add_argument('--biot-depth', type=int, default=4,
                       help='Number of BIOT transformer layers.')
    group.add_argument('--biot-heads', type=int, default=8,
                       help='Number of BIOT attention heads.')
    group.add_argument('--biot-n-fft', type=int, default=200,
                       help='STFT FFT size.')
    group.add_argument('--biot-hop-length', type=int, default=100,
                       help='STFT hop length.')


def apply_cli_args(config, args):
    config.biot_pretrained = args.biot_pretrained
    config.biot_target_fs = args.biot_target_fs
    config.biot_emb_size = args.biot_emb_size
    config.biot_depth = args.biot_depth
    config.biot_heads = args.biot_heads
    config.biot_n_fft = args.biot_n_fft
    config.biot_hop_length = args.biot_hop_length


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class BIOTEncoder(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_channels=18, n_fft=200, hop_length=100):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.projection = nn.Linear(n_fft // 2 + 1, emb_size)
        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.position = PositionalEncoding(emb_size)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=heads,
            dim_feedforward=4 * emb_size,
            dropout=0.2,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)

    def _stft(self, sample):
        if sample.shape[-1] < self.n_fft:
            sample = torch.nn.functional.pad(sample, (0, self.n_fft - sample.shape[-1]))
        spectral = torch.stft(
            sample.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)

    def forward(self, x):
        sequences = []
        max_channels = self.channel_tokens.num_embeddings
        for channel_idx in range(x.shape[1]):
            spec = self._stft(x[:, channel_idx:channel_idx + 1, :]).permute(0, 2, 1)
            emb = self.projection(spec)
            token = self.channel_tokens.weight[channel_idx % max_channels].view(1, 1, -1)
            sequences.append(self.position(emb + token))
        emb = torch.cat(sequences, dim=1)
        return self.transformer(emb).mean(dim=1)


class BIOTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_fs = getattr(config, 'fs', None)
        self.target_fs = getattr(config, 'biot_target_fs', 200)
        emb_size = getattr(config, 'biot_emb_size', 256)
        self.chan_conv = nn.Conv1d(getattr(config, 'CH', 2), 18, kernel_size=1, bias=False)
        self.biot = BIOTEncoder(
            emb_size=emb_size,
            heads=getattr(config, 'biot_heads', 8),
            depth=getattr(config, 'biot_depth', 4),
            n_channels=18,
            n_fft=getattr(config, 'biot_n_fft', 200),
            hop_length=getattr(config, 'biot_hop_length', 100),
        )
        self.classifier = nn.Sequential(nn.ELU(), nn.Linear(emb_size, getattr(config, 'num_classes', 2)))
        load_matching_state_dict(self, getattr(config, 'biot_pretrained', ''))

    def forward(self, x):
        x = maybe_resample(x, self.input_fs, self.target_fs)
        x = self.chan_conv(x)
        return self.classifier(self.biot(x))


def net(config):
    return BIOTClassifier(config)
