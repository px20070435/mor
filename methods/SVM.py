import os
import pickle

import numpy as np
from scipy import signal, stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from data_loader.data import Data


RMS_LOW_UV = 13.0
RMS_HIGH_UV = 150.0
EEG_BANDS = (
    ('delta', (1.0, 3.0)),
    ('theta', (4.0, 8.0)),
    ('alpha', (9.0, 13.0)),
    ('beta', (14.0, 20.0)),
)
HF_BAND = (40.0, 80.0)


def add_cli_args(parser):
    group = parser.add_argument_group('SVM method arguments')
    group.add_argument('--svm-c', type=float, default=1.0,
                       help='Regularization parameter for the SVM baseline.')
    group.add_argument('--svm-gamma', default='scale',
                       help='Kernel gamma for the SVM baseline, e.g. scale, auto, or a float string.')


def apply_cli_args(config, args):
    config.svm_c = args.svm_c
    try:
        config.svm_gamma = float(args.svm_gamma)
    except ValueError:
        config.svm_gamma = args.svm_gamma


def net(config):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=getattr(config, 'svm_c', 1.0),
            gamma=getattr(config, 'svm_gamma', 'scale'),
            probability=True,
            random_state=getattr(config, 'seed', 1),
        )),
    ])


def save_model(model, path):
    with open(path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def load_model(path):
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)


def extract_generator_dataset(generator, fs, verbose=True):
    data_segs = getattr(generator, 'data_segs', None)
    labels = np.asarray(getattr(generator, 'labels', []), dtype=np.uint8)

    if data_segs is None or isinstance(data_segs, dict):
        raise ValueError('SVM cached feature extraction only supports EEG generators with array-backed segments.')

    features = []
    valid_mask = []
    iterator = tqdm(data_segs, total=len(labels), disable=not verbose)
    for segment in iterator:
        feature_vector, is_valid = _extract_generator_segment_features(segment, fs)
        features.append(feature_vector)
        valid_mask.append(is_valid)

    if not features:
        return (
            np.empty((0, 42), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
            np.empty((0,), dtype=bool),
        )

    return (
        np.asarray(features, dtype=np.float32),
        labels.copy(),
        np.asarray(valid_mask, dtype=bool),
    )


def extract_dataset(config, recs, segments, verbose=True):
    features = []
    labels = []
    valid_mask = []

    prev_rec = None
    bandpassed = None
    raw_resampled = None

    for segment in tqdm(segments, disable=not verbose):
        rec_idx = int(segment[0])
        if rec_idx != prev_rec:
            bandpassed, raw_resampled = _load_recording(config, recs[rec_idx])
            prev_rec = rec_idx

        start = int(float(segment[1]) * config.fs)
        stop = int(float(segment[2]) * config.fs)
        feature_vector, is_valid = _extract_segment_features(
            bandpassed,
            raw_resampled,
            start,
            stop,
            config.fs,
        )
        features.append(feature_vector)
        labels.append(int(segment[3]))
        valid_mask.append(is_valid)

    if not features:
        return (
            np.empty((0, 42), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
            np.empty((0,), dtype=bool),
        )

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.uint8),
        np.asarray(valid_mask, dtype=bool),
    )


def _load_recording(config, rec):
    rec_data = Data.loadData(config.data_path, rec, modalities=['eeg'])

    idx_focal = [i for i, c in enumerate(rec_data.channels) if c == 'BTEleft SD']
    if not idx_focal:
        idx_focal = [i for i, c in enumerate(rec_data.channels) if c == 'BTEright SD']
    idx_cross = [i for i, c in enumerate(rec_data.channels) if c == 'CROSStop SD']
    if not idx_cross:
        idx_cross = [i for i, c in enumerate(rec_data.channels) if c == 'BTEright SD']

    focal_raw = _resample_channel(rec_data.data[idx_focal[0]], rec_data.fs[idx_focal[0]], config.fs)
    cross_raw = _resample_channel(rec_data.data[idx_cross[0]], rec_data.fs[idx_cross[0]], config.fs)

    return (
        (
            _bandpass_svm(focal_raw, config.fs),
            _bandpass_svm(cross_raw, config.fs),
        ),
        (
            focal_raw,
            cross_raw,
        ),
    )


def _extract_generator_segment_features(segment, fs):
    segment = np.asarray(segment, dtype=np.float32)
    if segment.ndim != 2:
        raise ValueError(f'Expected cached EEG segment with shape [time, channels], got {segment.shape!r}.')

    per_channel_features = []
    per_channel_validity = []
    n_channels = min(segment.shape[1], 2)

    for ch_idx in range(n_channels):
        raw_window = np.asarray(segment[:, ch_idx], dtype=np.float32)
        filtered_window = _bandpass_svm(raw_window, fs) if len(raw_window) else raw_window
        per_channel_features.extend(_extract_channel_features(filtered_window, raw_window, fs))
        per_channel_validity.append(_window_is_valid(filtered_window))

    if n_channels < 2:
        per_channel_features.extend(np.zeros(21 * (2 - n_channels), dtype=np.float32))
        per_channel_validity.extend([False] * (2 - n_channels))

    return np.asarray(per_channel_features, dtype=np.float32), bool(all(per_channel_validity))


def _resample_channel(ch_data, fs_data, fs_target):
    if fs_target != fs_data:
        ch_data = signal.resample(ch_data, int(fs_target * len(ch_data) / fs_data))
    return np.asarray(ch_data, dtype=np.float32)


def _bandpass_svm(ch_data, fs):
    b, a = signal.butter(4, [1.0 / (fs / 2), 25.0 / (fs / 2)], btype='bandpass')
    return signal.filtfilt(b, a, ch_data).astype(np.float32, copy=False)


def _extract_segment_features(bandpassed_recording, raw_recording, start, stop, fs):
    per_channel_features = []
    per_channel_validity = []

    for filtered_signal, raw_signal in zip(bandpassed_recording, raw_recording):
        filtered_window = _slice_with_padding(filtered_signal, start, stop)
        raw_window = _slice_with_padding(raw_signal, start, stop)
        per_channel_features.extend(_extract_channel_features(filtered_window, raw_window, fs))
        per_channel_validity.append(_window_is_valid(filtered_window))

    return np.asarray(per_channel_features, dtype=np.float32), bool(all(per_channel_validity))


def _slice_with_padding(signal_1d, start, stop):
    window_len = max(stop - start, 0)
    if window_len == 0:
        return np.zeros(0, dtype=np.float32)

    if start >= len(signal_1d):
        return np.zeros(window_len, dtype=np.float32)

    clipped = np.asarray(signal_1d[max(start, 0):min(stop, len(signal_1d))], dtype=np.float32)
    if len(clipped) == window_len:
        return clipped

    padded = np.zeros(window_len, dtype=np.float32)
    padded[:len(clipped)] = clipped
    return padded


def _window_is_valid(window):
    if len(window) == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float64)))
    return RMS_LOW_UV < rms < RMS_HIGH_UV


def _extract_channel_features(filtered_window, raw_window, fs):
    if len(filtered_window) == 0:
        return np.zeros(21, dtype=np.float32)

    filtered_window = np.asarray(filtered_window, dtype=np.float64)
    raw_window = np.asarray(raw_window, dtype=np.float64)

    zero_crossings = float(np.count_nonzero(np.diff(np.signbit(filtered_window))))
    maxima = float(len(signal.find_peaks(filtered_window)[0]))
    minima = float(len(signal.find_peaks(-filtered_window)[0]))
    skewness = float(np.nan_to_num(stats.skew(filtered_window), nan=0.0))
    kurt = float(np.nan_to_num(stats.kurtosis(filtered_window, fisher=False), nan=0.0))
    rms = float(np.sqrt(np.mean(np.square(filtered_window), dtype=np.float64)))

    freqs, psd = signal.welch(filtered_window, fs=fs, nperseg=min(len(filtered_window), fs))
    raw_freqs, raw_psd = signal.welch(raw_window, fs=fs, nperseg=min(len(raw_window), fs))
    total_power = _band_power(freqs, psd, (1.0, 25.0))
    total_raw_power = _band_power(raw_freqs, raw_psd, (1.0, HF_BAND[1]))

    peak_band = (freqs >= 1.0) & (freqs <= 25.0)
    if np.any(peak_band):
        peak_frequency = float(freqs[peak_band][np.argmax(psd[peak_band])])
    else:
        peak_frequency = 0.0

    mean_powers = []
    normalized_powers = []
    for _, band in EEG_BANDS:
        band_power = _band_power(freqs, psd, band)
        mean_powers.append(band_power)
        normalized_powers.append(band_power / (total_power + 1e-12))

    hf_power = _band_power(raw_freqs, raw_psd, HF_BAND)
    hf_power_norm = hf_power / (total_raw_power + 1e-12)

    coarse_window = filtered_window
    if len(coarse_window) > 100:
        coarse_window = signal.resample(coarse_window, 100)

    sample_entropy = _sample_entropy(coarse_window, m=2, r=0.2 * np.std(coarse_window))
    shannon_entropy = _shannon_entropy(filtered_window)
    spectral_entropy = _spectral_entropy(freqs, psd, band=(1.0, 25.0))

    features = [
        zero_crossings,
        maxima,
        minima,
        skewness,
        _safe_log1p(kurt),
        _safe_log1p(rms),
        _safe_log1p(total_power),
        peak_frequency,
        *[_safe_log1p(power) for power in mean_powers],
        *[_safe_log1p(power) for power in normalized_powers],
        _safe_log1p(hf_power),
        _safe_log1p(hf_power_norm),
        float(sample_entropy),
        _safe_log1p(shannon_entropy),
        float(spectral_entropy),
    ]
    return np.asarray(features, dtype=np.float32)


def _band_power(freqs, psd, band):
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _safe_log1p(value):
    return float(np.log1p(max(float(value), 0.0)))


def _sample_entropy(x, m=2, r=0.2):
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= m + 1 or not np.isfinite(r) or r <= 0:
        return 0.0

    emb_m = np.lib.stride_tricks.sliding_window_view(x, m)
    emb_m1 = np.lib.stride_tricks.sliding_window_view(x, m + 1)

    phi_m = _matching_probability(emb_m, r)
    phi_m1 = _matching_probability(emb_m1, r)
    if phi_m <= 0 or phi_m1 <= 0:
        return 0.0
    return float(-np.log(phi_m1 / phi_m))


def _matching_probability(embedded, tolerance):
    n = len(embedded)
    if n < 2:
        return 0.0

    distances = np.max(np.abs(embedded[:, None, :] - embedded[None, :, :]), axis=2)
    matches = np.sum(distances <= tolerance, axis=1) - 1
    return float(np.sum(matches) / (n * (n - 1)))


def _shannon_entropy(x, bins=32):
    hist, _ = np.histogram(x, bins=bins, density=False)
    prob = hist.astype(np.float64)
    prob = prob[prob > 0]
    if len(prob) == 0:
        return 0.0
    prob /= np.sum(prob)
    return float(-np.sum(prob * np.log2(prob)))


__all__ = [
    'add_cli_args',
    'apply_cli_args',
    'extract_dataset',
    'extract_generator_dataset',
    'load_model',
    'net',
    'save_model',
]


def _spectral_entropy(freqs, psd, band):
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    band_psd = psd[mask]
    if len(band_psd) == 0:
        return 0.0
    prob = band_psd.astype(np.float64)
    total = np.sum(prob)
    if total <= 0:
        return 0.0
    prob /= total
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))
