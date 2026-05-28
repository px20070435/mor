import csv
import copy
import gc
import os
import pickle
import sys
import time

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loader.key_generator import generate_data_keys_sequential, generate_data_keys_subsample, generate_data_keys_sequential_window
from data_loader.generator_ds import SegmentedGenerator, SequentialGenerator
from trains.routines import train_net, predict_net
from utils.eeg_utils import apply_preprocess_eeg, get_metrics_scoring, validation_metrics

from data_loader.data import Data

DATASET_DIR = os.path.join('data_loader', 'datasets')
GENERATOR_DIR = os.path.join('data_loader', 'generators')


def _limit_recs(recs, limit):
    if limit is None:
        return recs
    return recs[:max(0, int(limit))]


def _limit_segments(segments, limit):
    if limit is None:
        return segments
    return segments[:max(0, int(limit))]


def _get_model_loader(model_name):
    if model_name == 'ChronoNet':
        from methods.ChronoNet import net
    elif model_name == 'EEGnet':
        from methods.EEGnet import net
    elif model_name == 'DeepConvNet':
        from methods.DeepConv_Net import net
    elif model_name == 'SVM':
        from methods.SVM import net
    elif model_name == 'XGB':
        from methods.XGB import net
    elif model_name == 'STEEGFormer':
        from methods.STEEGFormer import net
    elif model_name == 'BIOT':
        from methods.BIOT import net
    elif model_name == 'BENDR':
        from methods.BENDR import net
    elif model_name == 'CBraMod':
        from methods.CBraMod import net
    elif model_name == 'Conformer':
        from methods.Conformer import net
    elif model_name == 'EEGPT':
        from methods.EEGPT import net
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return net


def _append_experiment_log(config, stage, payload):
    logs_dir = os.path.join(config.save_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, 'experiments.csv')
    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0
    with open(log_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow([
                'timestamp',
                'stage',
                'experiment',
                'model',
                'dataset',
                'sample_type',
                'factor',
                'frame',
                'stride',
                'stride_s',
                'fs',
                'channels',
                'save_dir',
                'summary',
            ])
        writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            stage,
            config.get_name(),
            config.model,
            config.dataset,
            config.sample_type,
            config.factor,
            config.frame,
            config.stride,
            config.stride_s,
            config.fs,
            config.CH,
            config.save_dir,
            str(payload),
        ])


def _load_cached_generator(path):
    # Older cached generator pickles may reference the legacy numpy._core path.
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)


def _limit_cached_generator(generator, limit):
    if limit is None:
        return generator

    limit = max(0, int(limit))
    if len(generator) <= limit:
        return generator

    limited = copy.copy(generator)
    data_segs = getattr(generator, 'data_segs', None)
    if isinstance(data_segs, dict):
        limited.data_segs = {key: value[:limit].copy() for key, value in data_segs.items()}
    elif data_segs is not None:
        limited.data_segs = data_segs[:limit].copy()
    limited.labels = np.asarray(generator.labels[:limit]).copy()
    return limited


def _sanitize_log_value(value):
    if isinstance(value, dict):
        return {key: _sanitize_log_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_log_value(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float('nan')
    return float(np.nanmean(arr))


def _nanmean_axis(values, axis):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([])
    valid = np.isfinite(arr)
    counts = np.sum(valid, axis=axis)
    sums = np.nansum(np.where(valid, arr, 0.0), axis=axis)
    return np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=float),
        where=counts > 0,
    )


def _trapz(y_values, x_values):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y_values, x_values)
    return np.trapz(y_values, x_values)


def _safe_auc(x_values, y_values, normalize_by=None):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if np.sum(valid) < 2:
        return float('nan')

    x_values = x_values[valid]
    y_values = y_values[valid]
    order = np.argsort(x_values)
    area = float(_trapz(y_values[order], x_values[order]))
    if normalize_by is not None and normalize_by > 0:
        area /= normalize_by
    return area


def _artifact_mask(config, rec):
    rec_data = Data.loadData(config.data_path, rec, modalities=['eeg'])
    ch_focal, ch_cross = apply_preprocess_eeg(config, rec_data)

    window = int(2 * config.fs)
    stride = int(1 * config.fs)
    rmsa_f = [
        np.sqrt(np.mean(ch_focal[start:start + window] ** 2))
        for start in range(0, len(ch_focal) - window + 1, stride)
    ]
    rmsa_c = [
        np.sqrt(np.mean(ch_cross[start:start + window] ** 2))
        for start in range(0, len(ch_cross) - window + 1, stride)
    ]
    mask_len = min(len(rmsa_f), len(rmsa_c))
    rmsa_f = np.asarray(rmsa_f[:mask_len], dtype=float)
    rmsa_c = np.asarray(rmsa_c[:mask_len], dtype=float)
    return (
        (rmsa_f > 13)
        & (rmsa_f < 150)
        & (rmsa_c > 13)
        & (rmsa_c < 150)
    )


def _summary_from_evaluation(
    thresholds,
    x_plot,
    sens_ovlp,
    prec_ovlp,
    fah_ovlp,
    f1_ovlp,
    sens_ovlp_plot,
    prec_ovlp_plot,
    sens_epoch,
    spec_epoch,
    prec_epoch,
    fah_epoch,
    f1_epoch,
    score,
):
    thresholds = np.asarray(thresholds, dtype=float)
    th_05_idx = int(np.argmin(np.abs(thresholds - 0.5)))

    sens_ovlp_arr = np.asarray(sens_ovlp, dtype=float)
    prec_ovlp_arr = np.asarray(prec_ovlp, dtype=float)
    fah_ovlp_arr = np.asarray(fah_ovlp, dtype=float)
    f1_ovlp_arr = np.asarray(f1_ovlp, dtype=float)
    sens_epoch_arr = np.asarray(sens_epoch, dtype=float)
    spec_epoch_arr = np.asarray(spec_epoch, dtype=float)
    prec_epoch_arr = np.asarray(prec_epoch, dtype=float)
    fah_epoch_arr = np.asarray(fah_epoch, dtype=float)
    f1_epoch_arr = np.asarray(f1_epoch, dtype=float)
    score_arr = np.asarray(score, dtype=float)

    mean_sens_epoch = _nanmean_axis(sens_epoch_arr, axis=0) if sens_epoch_arr.size else np.array([])
    mean_spec_epoch = _nanmean_axis(spec_epoch_arr, axis=0) if spec_epoch_arr.size else np.array([])
    mean_prec_epoch = _nanmean_axis(prec_epoch_arr, axis=0) if prec_epoch_arr.size else np.array([])
    mean_sens_ovlp_plot = _nanmean_axis(sens_ovlp_plot, axis=0) if sens_ovlp_plot else np.array([])

    fpr_epoch = 1.0 - mean_spec_epoch
    auroc = _safe_auc(fpr_epoch, mean_sens_epoch)
    aupr = _safe_auc(mean_sens_epoch, mean_prec_epoch)
    ausf = _safe_auc(x_plot, mean_sens_ovlp_plot, normalize_by=float(np.max(x_plot))) if len(mean_sens_ovlp_plot) else float('nan')

    return {
        'threshold': float(thresholds[th_05_idx]),
        'sens_ovlp': _safe_nanmean(sens_ovlp_arr[:, th_05_idx]) if sens_ovlp_arr.size else float('nan'),
        'prec_ovlp': _safe_nanmean(prec_ovlp_arr[:, th_05_idx]) if prec_ovlp_arr.size else float('nan'),
        'fah_ovlp': _safe_nanmean(fah_ovlp_arr[:, th_05_idx]) if fah_ovlp_arr.size else float('nan'),
        'f1_ovlp': _safe_nanmean(f1_ovlp_arr[:, th_05_idx]) if f1_ovlp_arr.size else float('nan'),
        'sens_epoch': _safe_nanmean(sens_epoch_arr[:, th_05_idx]) if sens_epoch_arr.size else float('nan'),
        'spec_epoch': _safe_nanmean(spec_epoch_arr[:, th_05_idx]) if spec_epoch_arr.size else float('nan'),
        'prec_epoch': _safe_nanmean(prec_epoch_arr[:, th_05_idx]) if prec_epoch_arr.size else float('nan'),
        'fah_epoch': _safe_nanmean(fah_epoch_arr[:, th_05_idx]) if fah_epoch_arr.size else float('nan'),
        'f1_epoch': _safe_nanmean(f1_epoch_arr[:, th_05_idx]) if f1_epoch_arr.size else float('nan'),
        'score': _safe_nanmean(score_arr[:, th_05_idx]) if score_arr.size else float('nan'),
        'auroc': auroc,
        'aupr': aupr,
        'ausf': ausf,
    }


def _write_evaluation_summary_csv(summary_file, config, summary):
    fieldnames = [
        'experiment',
        'model',
        'threshold',
        'sens',
        'fa_h',
        'auroc',
        'aupr',
        'ausf',
        'sens_epoch',
        'spec_epoch',
        'prec_epoch',
        'fah_epoch',
        'f1_epoch',
        'sens_ovlp',
        'prec_ovlp',
        'fah_ovlp',
        'f1_ovlp',
        'score',
    ]
    with open(summary_file, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'experiment': config.get_name(),
            'model': config.model,
            'threshold': summary['threshold'],
            'sens': summary['sens_ovlp'],
            'fa_h': summary['fah_ovlp'],
            'auroc': summary['auroc'],
            'aupr': summary['aupr'],
            'ausf': summary['ausf'],
            'sens_epoch': summary['sens_epoch'],
            'spec_epoch': summary['spec_epoch'],
            'prec_epoch': summary['prec_epoch'],
            'fah_epoch': summary['fah_epoch'],
            'f1_epoch': summary['f1_epoch'],
            'sens_ovlp': summary['sens_ovlp'],
            'prec_ovlp': summary['prec_ovlp'],
            'fah_ovlp': summary['fah_ovlp'],
            'f1_ovlp': summary['f1_ovlp'],
            'score': summary['score'],
        })


def train(config, load_generators, save_generators):
    """ Routine to run the model's training routine.

        Args:
            config (cls): a config object with the data input type and model parameters
            load_generators (bool): boolean to load the training and validation generators from file
            save_generators (bool): boolean to save the training and validation generators
    """

    name = config.get_name()

    net = _get_model_loader(config.model)

    if not os.path.exists(os.path.join(config.save_dir, 'models')):
        os.mkdir(os.path.join(config.save_dir, 'models'))

    model_save_path = os.path.join(config.save_dir, 'models', name)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    config_path = os.path.join(config.save_dir, 'models', name, 'configs')
    if not os.path.exists(config_path):
        os.mkdir(config_path)

    config.save_config(save_path=config_path)

    #######################################################################################################################
    ### Fixed train/val/test ###
    #######################################################################################################################
    if config.cross_validation == 'fixed':
        
        if config.dataset == 'SZ2':

            train_pats_list = pd.read_csv(os.path.join(DATASET_DIR, 'SZ2_training.tsv'), sep = '\t', header = None, skiprows = [0,1,2])
            train_pats_list = train_pats_list[0].to_list()
            train_recs_list = [[s, r.split('_')[-2]] for s in train_pats_list for r in os.listdir(os.path.join(config.data_path, s, 'ses-01', 'eeg')) if 'edf' in r]
            train_recs_list = _limit_recs(train_recs_list, getattr(config, 'max_train_recordings', None))

            if load_generators:
                print('Loading generators...')
                generator_cache_name = config.dataset + '_frame-' + str(config.frame) + '_sampletype-' + config.sample_type
                gen_train = _load_cached_generator(os.path.join(GENERATOR_DIR, 'gen_train_' + generator_cache_name + '.pkl'))
                gen_val = _load_cached_generator(os.path.join(GENERATOR_DIR, 'gen_val.pkl'))
                gen_train = _limit_cached_generator(gen_train, getattr(config, 'max_train_segments', None))
                gen_val = _limit_cached_generator(gen_val, getattr(config, 'max_val_segments', None))

            else:
                if config.sample_type == 'subsample':
                    train_segments = generate_data_keys_subsample(config, train_recs_list)
                    train_segments = _limit_segments(train_segments, getattr(config, 'max_train_segments', None))

                val_pats_list = pd.read_csv(os.path.join(DATASET_DIR, 'SZ2_validation.tsv'), sep = '\t', header = None, skiprows = [0,1,2])
                val_pats_list = val_pats_list[0].to_list()
                val_recs_list = [[s, r.split('_')[-2]] for s in val_pats_list for r in os.listdir(os.path.join(config.data_path, s, 'ses-01', 'eeg')) if 'edf' in r]
                val_recs_list = _limit_recs(val_recs_list, getattr(config, 'max_val_recordings', None))
                val_segments = generate_data_keys_sequential_window(config, val_recs_list, 5*60)
                val_segments = _limit_segments(val_segments, getattr(config, 'max_val_segments', None))

                if config.model not in ('SVM', 'XGB'):
                    print('Generating training segments...')
                    gen_train = SegmentedGenerator(config, train_recs_list, train_segments, batch_size=config.batch_size, shuffle=True)

                    if save_generators:
                        generator_cache_name = config.dataset + '_frame-' + str(config.frame) + '_sampletype-' + config.sample_type
                        if not os.path.exists(GENERATOR_DIR):
                            os.mkdir(GENERATOR_DIR)

                        with open(os.path.join(GENERATOR_DIR, 'gen_train_' + generator_cache_name + '.pkl'), 'wb') as outp:
                            pickle.dump(gen_train, outp, pickle.HIGHEST_PROTOCOL)

                    print('Generating validation segments...')
                    gen_val = SequentialGenerator(config, val_recs_list, val_segments, batch_size=600, shuffle=False)

                    if save_generators:
                        with open(os.path.join(GENERATOR_DIR, 'gen_val.pkl'), 'wb') as outp:
                            pickle.dump(gen_val, outp, pickle.HIGHEST_PROTOCOL)

            print('### Training model....')
            config.save_config(save_path=config_path)
            start_train = time.time()

            if config.model in ('SVM', 'XGB'):
                if config.model == 'SVM':
                    from methods.SVM import extract_dataset, extract_generator_dataset, save_model
                else:
                    from methods.XGB import extract_dataset, extract_generator_dataset, save_model

                if load_generators:
                    print(f'Extracting {config.model} training features from cached generators...')
                    x_train, y_train, train_valid = extract_generator_dataset(gen_train, config.fs, verbose=True)
                    print(f'Extracting {config.model} validation features from cached generators...')
                    x_val, y_val, val_valid = extract_generator_dataset(gen_val, config.fs, verbose=True)
                    train_segment_count = len(gen_train)
                    val_segment_count = len(gen_val)
                else:
                    print(f'Extracting {config.model} training features...')
                    x_train, y_train, train_valid = extract_dataset(config, train_recs_list, train_segments, verbose=True)
                    print(f'Extracting {config.model} validation features...')
                    x_val, y_val, val_valid = extract_dataset(config, val_recs_list, val_segments, verbose=True)
                    train_segment_count = len(train_segments)
                    val_segment_count = len(val_segments)

                x_train = x_train[train_valid]
                y_train = y_train[train_valid]
                if len(x_train) == 0:
                    raise ValueError(
                        f'No valid {config.model} training segments remained after RMS filtering. '
                        'Try using more recordings or the full training split.'
                    )

                x_val = x_val[val_valid]
                y_val = y_val[val_valid]

                model = net(config)
                model.fit(x_train, y_train)
                if len(x_val):
                    val_prob = model.predict_proba(x_val)[:, 1]
                    val_metrics = validation_metrics(y_val, val_prob)
                else:
                    val_prob = np.array([], dtype=np.float32)
                    val_metrics = {
                        'accuracy': float('nan'),
                        'sens': float('nan'),
                        'spec': float('nan'),
                        'sens_ovlp': float('nan'),
                        'fah_ovlp': float('nan'),
                        'fah_epoch': float('nan'),
                        'faRate_epoch': float('nan'),
                        'score': float('nan'),
                        'auc': float('nan'),
                    }

                weights_dir = os.path.join(model_save_path, 'Weights')
                history_dir = os.path.join(model_save_path, 'History')
                os.makedirs(weights_dir, exist_ok=True)
                os.makedirs(history_dir, exist_ok=True)

                save_model(model, os.path.join(weights_dir, name + '.pkl'))
                with open(os.path.join(history_dir, name + '.csv'), 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['split', 'samples', 'valid_samples', 'accuracy', 'sens', 'spec', 'sens_ovlp', 'fah_ovlp', 'fah_epoch', 'faRate_epoch', 'score', 'auc'])
                    writer.writerow([
                        'validation',
                        val_segment_count,
                        int(np.sum(val_valid)),
                        val_metrics['accuracy'],
                        val_metrics['sens'],
                        val_metrics['spec'],
                        val_metrics['sens_ovlp'],
                        val_metrics['fah_ovlp'],
                        val_metrics['fah_epoch'],
                        val_metrics['faRate_epoch'],
                        val_metrics['score'],
                        val_metrics['auc'],
                    ])
                _append_experiment_log(config, 'train', {
                    'train_segments': train_segment_count,
                    'train_valid_segments': int(np.sum(train_valid)),
                    'val_segments': val_segment_count,
                    'val_valid_segments': int(np.sum(val_valid)),
                    'used_cached_generators': bool(load_generators),
                    'val_metrics': _sanitize_log_value(val_metrics),
                })
                print(f"{config.model} validation score: {val_metrics['score']:.4f} | auc: {val_metrics['auc']:.4f}")
            else:
                model = net(config)
                train_net(config, model, gen_train, gen_val, model_save_path)
            
            end_train = time.time() - start_train
            print('Total train duration = ', end_train / 60)


#######################################################################################################################
#######################################################################################################################


def predict(config):

    name = config.get_name()

    model_save_path = os.path.join(config.save_dir, 'models', name)

    if not os.path.exists(os.path.join(config.save_dir, 'predictions')):
        os.mkdir(os.path.join(config.save_dir, 'predictions'))
    if not os.path.exists(os.path.join(config.save_dir, 'predictions', name)):
        os.mkdir(os.path.join(config.save_dir, 'predictions', name))

    test_pats_list = pd.read_csv(os.path.join(DATASET_DIR, config.dataset + '_test.tsv'), sep = '\t', header = None, skiprows = [0,1,2])
    test_pats_list = test_pats_list[0].to_list()
    test_recs_list = [[s, r.split('_')[-2]] for s in test_pats_list for r in os.listdir(os.path.join(config.data_path, s, 'ses-01', 'eeg')) if 'edf' in r]
    test_recs_list = _limit_recs(test_recs_list, getattr(config, 'max_test_recordings', None))

    model_weights_path = os.path.join(model_save_path, 'Weights', name + '.pt')
    max_test_segments = getattr(config, 'max_test_segments', None)

    config.load_config(config_path=os.path.join(model_save_path, 'configs'), config_name=name+'.cfg')
    config.max_test_segments = max_test_segments
        
    net = _get_model_loader(config.model)

    for rec in tqdm(test_recs_list):
        if os.path.isfile(os.path.join(config.save_dir, 'predictions', name, rec[0] + '_' + rec[1] + '_preds.h5')):
            print(rec[0] + ' ' + rec[1] + ' exists. Skipping...')
        else:
            segments = generate_data_keys_sequential(config, [rec], verbose=False)
            segments = _limit_segments(segments, getattr(config, 'max_test_segments', None))
            if config.model in ('SVM', 'XGB'):
                if config.model == 'SVM':
                    from methods.SVM import extract_dataset, load_model
                else:
                    from methods.XGB import extract_dataset, load_model

                model_weights_path = os.path.join(model_save_path, 'Weights', name + '.pkl')
                model = load_model(model_weights_path)
                x_test, y_true, valid_mask = extract_dataset(config, [rec], segments, verbose=False)
                y_pred = np.zeros(len(segments), dtype=np.float32)
                if np.any(valid_mask):
                    y_pred[valid_mask] = model.predict_proba(x_test[valid_mask])[:, 1].astype(np.float32)
            else:
                gen_test = SequentialGenerator(config, [rec], segments, batch_size=len(segments), shuffle=False, verbose=False)
                model = net(config)
                y_pred, y_true = predict_net(gen_test, model_weights_path, model, batch_size=len(segments), device='cpu')

            with h5py.File(os.path.join(config.save_dir, 'predictions', name, rec[0] + '_' + rec[1] + '_preds.h5'), 'w') as f:
                f.create_dataset('y_pred', data=y_pred)
                f.create_dataset('y_true', data=y_true)

            gc.collect()

   
#######################################################################################################################
#######################################################################################################################


def evaluate(config):

    name = config.get_name()

    pred_path = os.path.join(config.save_dir, 'predictions', name)
    pred_fs = 1

    thresholds = list(np.around(np.linspace(0,1,51),2))

    x_plot = np.linspace(0, 200, 200)

    if not os.path.exists(os.path.join(config.save_dir, 'results')):
        os.mkdir(os.path.join(config.save_dir, 'results'))

    result_file = os.path.join(config.save_dir, 'results', name + '.h5')
    summary_file = os.path.join(config.save_dir, 'results', name + '_summary.csv')

    sens_ovlp = []
    prec_ovlp = []
    fah_ovlp = []
    sens_ovlp_plot = []
    prec_ovlp_plot = []
    f1_ovlp = []

    sens_epoch = []
    spec_epoch = []
    prec_epoch = []
    fah_epoch = []
    f1_epoch = []

    score = []

    pred_files = [x for x in os.listdir(pred_path)]
    pred_files.sort()

    for file in tqdm(pred_files):
        with h5py.File(os.path.join(pred_path, file), 'r') as f:
            y_pred = list(f['y_pred'])
            y_true = list(f['y_true'])

        sens_ovlp_th = []
        prec_ovlp_th = []
        fah_ovlp_th = []
        f1_ovlp_th = []

        sens_epoch_th = []
        spec_epoch_th = []
        prec_epoch_th = []
        fah_epoch_th = []
        f1_epoch_th = []

        score_th = []

        rec = [file.split('_')[0], file.split('_')[1]]

        rmsa = _artifact_mask(config, rec)
        if len(y_pred) != len(rmsa):
            min_len = min(len(y_pred), len(rmsa))
            y_pred = y_pred[:min_len]
            y_true = y_true[:min_len]
            rmsa = rmsa[:min_len]
        y_pred = np.where(rmsa == 0, 0, y_pred)

        for th in thresholds:
            sens_ovlp_rec, prec_ovlp_rec, FA_ovlp_rec, f1_ovlp_rec, sens_epoch_rec, spec_epoch_rec, prec_epoch_rec, FA_epoch_rec, f1_epoch_rec = get_metrics_scoring(y_pred, y_true, pred_fs, th)

            sens_ovlp_th.append(sens_ovlp_rec)
            prec_ovlp_th.append(prec_ovlp_rec)
            fah_ovlp_th.append(FA_ovlp_rec)
            f1_ovlp_th.append(f1_ovlp_rec)
            sens_epoch_th.append(sens_epoch_rec)
            spec_epoch_th.append(spec_epoch_rec)
            prec_epoch_th.append(prec_epoch_rec)
            fah_epoch_th.append(FA_epoch_rec)
            f1_epoch_th.append(f1_epoch_rec)
            score_th.append(sens_ovlp_rec*100-0.4*FA_epoch_rec)

        sens_ovlp.append(sens_ovlp_th)
        prec_ovlp.append(prec_ovlp_th)
        fah_ovlp.append(fah_ovlp_th)
        f1_ovlp.append(f1_ovlp_th)

        sens_epoch.append(sens_epoch_th)
        spec_epoch.append(spec_epoch_th)
        prec_epoch.append(prec_epoch_th)
        fah_epoch.append(fah_epoch_th)
        f1_epoch.append(f1_epoch_th)

        score.append(score_th)

        to_cut = np.argmax(fah_ovlp_th)
        fah_ovlp_plot_rec = fah_ovlp_th[to_cut:]
        sens_ovlp_plot_rec = sens_ovlp_th[to_cut:]
        prec_ovlp_plot_rec = prec_ovlp_th[to_cut:]

        y_plot = np.interp(x_plot, fah_ovlp_plot_rec[::-1], sens_ovlp_plot_rec[::-1])
        sens_ovlp_plot.append(y_plot)
        y_plot = np.interp(x_plot, sens_ovlp_plot_rec[::-1], prec_ovlp_plot_rec[::-1])
        prec_ovlp_plot.append(y_plot)

    score_05 = [x[25] for x in score]
    score_05_mean = _safe_nanmean(score_05)
    summary = _summary_from_evaluation(
        thresholds,
        x_plot,
        sens_ovlp,
        prec_ovlp,
        fah_ovlp,
        f1_ovlp,
        sens_ovlp_plot,
        prec_ovlp_plot,
        sens_epoch,
        spec_epoch,
        prec_epoch,
        fah_epoch,
        f1_epoch,
        score,
    )

    print('Score: ' + ("nan" if np.isnan(score_05_mean) else f"{score_05_mean:.2f}"))
    print(
        "Evaluation summary at threshold "
        f"{summary['threshold']:.2f}: "
        f"Sens={summary['sens_ovlp']:.4f}, "
        f"FA/h={summary['fah_ovlp']:.4f}, "
        f"AUROC={summary['auroc']:.4f}, "
        f"AUPR={summary['aupr']:.4f}, "
        f"AUSF={summary['ausf']:.4f}"
    )

    with h5py.File(result_file, 'w') as f:
        f.create_dataset('thresholds', data=thresholds)
        f.create_dataset('sens_ovlp', data=sens_ovlp)
        f.create_dataset('prec_ovlp', data=prec_ovlp)
        f.create_dataset('fah_ovlp', data=fah_ovlp)
        f.create_dataset('f1_ovlp', data=f1_ovlp)
        f.create_dataset('sens_ovlp_plot', data=sens_ovlp_plot)
        f.create_dataset('prec_ovlp_plot', data=prec_ovlp_plot)
        f.create_dataset('x_plot', data=x_plot)
        f.create_dataset('sens_epoch', data=sens_epoch)
        f.create_dataset('spec_epoch', data=spec_epoch)
        f.create_dataset('prec_epoch', data=prec_epoch)
        f.create_dataset('fah_epoch', data=fah_epoch)
        f.create_dataset('f1_epoch', data=f1_epoch)
        f.create_dataset('score', data=score)
        summary_group = f.create_group('summary')
        for key, value in summary.items():
            summary_group.create_dataset(key, data=value)

    _write_evaluation_summary_csv(summary_file, config, summary)

    _append_experiment_log(config, 'evaluate', {
        'result_file': result_file,
        'summary_file': summary_file,
        'score_at_0.5_mean': score_05_mean,
        'summary': _sanitize_log_value(summary),
        'prediction_files': len(pred_files),
    })

