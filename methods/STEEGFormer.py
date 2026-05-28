import math
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

try:
    import timm.models.vision_transformer
except ImportError as exc:
    timm = None
    _TIMM_IMPORT_ERROR = exc
else:
    _TIMM_IMPORT_ERROR = None


DEFAULT_SEIZEIT2_CHANNELS = ('BTEleft SD', 'CROSStop SD')
DEFAULT_SEIZEIT2_CHANNEL_INDICES = (143, 144)


def add_cli_args(parser):
    group = parser.add_argument_group('ST-EEGFormer method arguments')
    group.add_argument('--steegformer-variant', default='small',
                       choices=('small', 'base', 'large'),
                       help='ST-EEGFormer backbone size.')
    group.add_argument('--steegformer-pretrained', default='',
                       help='Optional ST-EEGFormer checkpoint path.')
    group.add_argument('--steegformer-freeze-backbone', action='store_true',
                       help='Train only the classification head.')
    group.add_argument('--steegformer-patch-size', type=int, default=16,
                       help='Temporal patch size in samples.')
    group.add_argument('--steegformer-target-fs', type=int, default=128,
                       help='Sampling frequency expected inside the ST-EEGFormer adapter.')
    group.add_argument('--steegformer-channel-indices', default='143,144',
                       help='Comma-separated channel embedding indices for the two SeizeIT2 channels.')
    group.add_argument('--steegformer-global-pool', default='avg',
                       choices=('avg', 'cls'),
                       help='Pooling strategy for the transformer output.')


def apply_cli_args(config, args):
    config.steegformer_variant = args.steegformer_variant
    config.steegformer_pretrained = args.steegformer_pretrained
    config.steegformer_freeze_backbone = args.steegformer_freeze_backbone
    config.steegformer_patch_size = args.steegformer_patch_size
    config.steegformer_target_fs = args.steegformer_target_fs
    config.steegformer_channel_indices = _parse_channel_indices(args.steegformer_channel_indices)
    config.steegformer_global_pool = args.steegformer_global_pool


def _parse_channel_indices(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if value is None or str(value).strip() == '':
        return list(DEFAULT_SEIZEIT2_CHANNEL_INDICES)
    return [int(v.strip()) for v in str(value).split(',') if v.strip()]


class PatchEmbedEEG(nn.Module):
    """ST-EEGFormer temporal patch embedding for EEG shaped as [batch, channels, time]."""

    def __init__(self, patch_size=16, embed_dim=512):
        super().__init__()
        self.p = patch_size
        self.embed_dim = embed_dim
        self.unfold = nn.Unfold(kernel_size=(1, patch_size), stride=patch_size)
        self.proj = nn.Linear(self.p, self.embed_dim)

    def forward(self, x):
        patches = self.patchify_eeg(x)
        return self.proj(patches)

    def patchify_eeg(self, x):
        batch_size, channels, _ = x.shape
        unfolded = self.unfold(x.unsqueeze(2))
        _, _, seq = unfolded.shape
        unfolded = unfolded.reshape(batch_size, channels, self.p, seq)
        return unfolded.permute(0, 3, 1, 2)


class ChannelPositionalEmbed(nn.Module):
    def __init__(self, embedding_dim, max_channels=145):
        super().__init__()
        self.channel_transformation = nn.Embedding(max_channels, embedding_dim)
        init.zeros_(self.channel_transformation.weight)

    def forward(self, channel_indices):
        return self.channel_transformation(channel_indices)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float())
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position.float() * div_term)
        pe[0, :, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)

    def get_cls_token(self):
        return self.pe[0, 0, :]

    def forward(self, seq_indices):
        batch_size, seq_len = seq_indices.shape
        if int(seq_indices.max()) >= self.pe.shape[1]:
            raise ValueError(
                f'ST-EEGFormer sequence index {int(seq_indices.max())} exceeds '
                f'temporal positional capacity {self.pe.shape[1]}.'
            )
        return self.pe[0, seq_indices.reshape(-1)].reshape(batch_size, seq_len, -1)


class VisionTransformerEEG(timm.models.vision_transformer.VisionTransformer if _TIMM_IMPORT_ERROR is None else nn.Module):
    """Minimal ST-EEGFormer ViT head adapted from the upstream implementation."""

    def __init__(self, global_pool=False, **kwargs):
        if _TIMM_IMPORT_ERROR is not None:
            raise ImportError('ST-EEGFormer requires the timm package. Install requirements.txt first.') from _TIMM_IMPORT_ERROR

        super().__init__(**kwargs)
        self.global_pool = bool(global_pool)
        if self.global_pool:
            self.fc_norm = kwargs['norm_layer'](kwargs['embed_dim'])
        elif not hasattr(self, 'norm'):
            self.norm = kwargs['norm_layer'](kwargs['embed_dim'])

        self.patch_embed = PatchEmbedEEG(patch_size=kwargs['patch_size'], embed_dim=kwargs['embed_dim'])
        self.enc_channel_emd = ChannelPositionalEmbed(kwargs['embed_dim'])
        self.enc_temporal_emd = TemporalPositionalEncoding(kwargs['embed_dim'], max_len=512)
        if not hasattr(self, 'head_drop'):
            self.head_drop = nn.Identity()

    def forward_features(self, eeg, chan_idx):
        batch_size = eeg.shape[0]
        x = self.patch_embed(eeg)
        _, seq, channels, d_model = x.shape
        seq_total = seq * channels
        x = x.reshape(batch_size, seq_total, d_model)

        eeg_chan_indices = chan_idx.unsqueeze(1).repeat(1, seq, 1).reshape(batch_size, seq_total)
        seq_tensor = torch.arange(1, seq + 1, device=eeg.device)
        eeg_seq_indices = seq_tensor.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, channels).reshape(batch_size, seq_total)

        x = x + self.enc_temporal_emd(eeg_seq_indices) + self.enc_channel_emd(eeg_chan_indices)
        cls_token = self.cls_token + self.enc_temporal_emd.get_cls_token()
        cls_tokens = cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        if self.global_pool:
            return self.fc_norm(x[:, 1:, :].mean(dim=1))
        x = self.norm(x)
        return x[:, 0]

    def forward(self, eeg, chan_idx):
        x = self.forward_features(eeg, chan_idx)
        return self.head(self.head_drop(x))


def _make_backbone(config):
    variant = getattr(config, 'steegformer_variant', 'small')
    patch_size = getattr(config, 'steegformer_patch_size', 16)
    global_pool = getattr(config, 'steegformer_global_pool', 'avg') == 'avg'
    common = {
        'num_classes': getattr(config, 'num_classes', 2),
        'patch_size': patch_size,
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'global_pool': global_pool,
    }
    if variant == 'small':
        return VisionTransformerEEG(embed_dim=512, depth=8, num_heads=8, mlp_ratio=4, **common)
    if variant == 'base':
        return VisionTransformerEEG(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **common)
    if variant == 'large':
        return VisionTransformerEEG(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, **common)
    raise ValueError(f'Unsupported ST-EEGFormer variant: {variant}')


class STEEGFormerClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_fs = getattr(config, 'fs', None)
        self.target_fs = getattr(config, 'steegformer_target_fs', 128)
        self.model = _make_backbone(config)

        channel_indices = _parse_channel_indices(getattr(config, 'steegformer_channel_indices', DEFAULT_SEIZEIT2_CHANNEL_INDICES))
        if len(channel_indices) != getattr(config, 'CH', len(channel_indices)):
            raise ValueError(
                'ST-EEGFormer channel index count must match config.CH. '
                f'Got {len(channel_indices)} indices for CH={getattr(config, "CH", None)}.'
            )
        self.register_buffer('channel_indices', torch.tensor(channel_indices, dtype=torch.long), persistent=False)

        pretrained = getattr(config, 'steegformer_pretrained', '')
        if pretrained:
            self._load_pretrained(pretrained)

        if getattr(config, 'steegformer_freeze_backbone', False):
            self._freeze_backbone()

    def _load_pretrained(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'ST-EEGFormer checkpoint not found: {checkpoint_path}')

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            checkpoint_model = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
        else:
            checkpoint_model = checkpoint

        state_dict = self.model.state_dict()
        filtered = {}
        skipped = []
        for key, value in checkpoint_model.items():
            clean_key = key.removeprefix('module.')
            if clean_key in state_dict and state_dict[clean_key].shape == value.shape:
                filtered[clean_key] = value
            else:
                skipped.append(clean_key)

        msg = self.model.load_state_dict(filtered, strict=False)
        if hasattr(self.model.head, 'weight'):
            init.trunc_normal_(self.model.head.weight, std=2e-5)
        if hasattr(self.model.head, 'bias') and self.model.head.bias is not None:
            init.zeros_(self.model.head.bias)
        print(
            f'Loaded ST-EEGFormer checkpoint from {checkpoint_path}; '
            f'used {len(filtered)} tensors, skipped {len(skipped)} tensors. '
            f'Missing keys: {len(msg.missing_keys)}, unexpected keys: {len(msg.unexpected_keys)}.'
        )

    def _freeze_backbone(self):
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = name.startswith('head.')

    def _resample_if_needed(self, x):
        if self.input_fs is None or self.target_fs is None or int(self.input_fs) == int(self.target_fs):
            return x
        target_len = max(1, int(round(x.shape[-1] * float(self.target_fs) / float(self.input_fs))))
        return F.interpolate(x, size=target_len, mode='linear', align_corners=False)

    def forward(self, x):
        x = self._resample_if_needed(x)
        chan_idx = self.channel_indices.to(x.device).unsqueeze(0).expand(x.shape[0], -1)
        return self.model(x, chan_idx)


def net(config):
    return STEEGFormerClassifier(config)
