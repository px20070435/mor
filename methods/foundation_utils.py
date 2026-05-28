import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def maybe_resample(x, input_fs, target_fs):
    if input_fs is None or target_fs is None or int(input_fs) == int(target_fs):
        return x
    target_len = max(1, int(round(x.shape[-1] * float(target_fs) / float(input_fs))))
    return F.interpolate(x, size=target_len, mode='linear', align_corners=False)


def load_matching_state_dict(module, checkpoint_path, keys=('model', 'state_dict'), strip_prefixes=('module.',)):
    if not checkpoint_path:
        return None
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in keys:
            if key in checkpoint:
                state = checkpoint[key]
                break

    current = module.state_dict()
    filtered = {}
    skipped = 0
    for key, value in state.items():
        clean_key = key
        for prefix in strip_prefixes:
            clean_key = clean_key.removeprefix(prefix)
        if clean_key in current and current[clean_key].shape == value.shape:
            filtered[clean_key] = value
        else:
            skipped += 1

    msg = module.load_state_dict(filtered, strict=False)
    print(
        f'Loaded {checkpoint_path}: used {len(filtered)} tensors, skipped {skipped}; '
        f'missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}.'
    )
    return msg


class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.net(x)
