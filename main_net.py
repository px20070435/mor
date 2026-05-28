import argparse
import os
import random

import numpy as np
import torch

from data_loader import key_generator
from methods import BENDR, BIOT, CBraMod, Conformer, EEGPT, STEEGFormer, SVM, XGB
from trains import main_func
from trains.config import Config


METHOD_CHOICES = [
    'ChronoNet', 'EEGnet', 'DeepConvNet', 'SVM', 'XGB',
    'STEEGFormer', 'BIOT', 'BENDR', 'CBraMod', 'Conformer', 'EEGPT',
]
METHOD_CONFIGURERS = {
    'SVM': SVM.apply_cli_args,
    'XGB': XGB.apply_cli_args,
    'STEEGFormer': STEEGFormer.apply_cli_args,
    'BIOT': BIOT.apply_cli_args,
    'BENDR': BENDR.apply_cli_args,
    'CBraMod': CBraMod.apply_cli_args,
    'Conformer': Conformer.apply_cli_args,
    'EEGPT': EEGPT.apply_cli_args,
}


def add_experiment_args(parser):
    group = parser.add_argument_group('experiment arguments')
    group.add_argument('--model', default='ChronoNet', choices=METHOD_CHOICES,
                       help='Model architecture to train.')
    group.add_argument('--dataset', default='SZ2',
                       help='Dataset split prefix under data_loader/datasets.')
    group.add_argument('--save-dir', default='net/save_dir',
                       help='Directory for checkpoints, predictions, histories, and results.')


def add_data_args(parser):
    group = parser.add_argument_group('data and preprocessing arguments')
    group.add_argument('--data-path', default='/data1/zhihao/SeizeIT2/ds005873-1.1.0/ds005873',
                       help='Path to the BIDS dataset root.')
    group.add_argument('--fs', type=int, default=250,
                       help='Sampling frequency after preprocessing.')
    group.add_argument('--channels', type=int, default=2,
                       help='Number of EEG channels.')
    group.add_argument('--frame', type=float, default=2,
                       help='Input window size in seconds.')
    group.add_argument('--stride', type=float, default=1,
                       help='Stride between background EEG segments in seconds.')
    group.add_argument('--stride-s', type=float, default=0.5,
                       help='Stride between seizure EEG segments in seconds.')
    group.add_argument('--boundary', type=float, default=0.5,
                       help='Minimum seizure proportion in a window for the positive class.')
    group.add_argument('--sample-type', default='subsample',
                       help='Training segment sampling method.')
    group.add_argument('--factor', type=int, default=5,
                       help='Balancing factor between negative and positive segments.')
    group.add_argument('--cross-validation', default='fixed',
                       help='Validation strategy.')
    group.add_argument('--max-train-recordings', type=int, default=None,
                       help='Optional cap on the number of training recordings to load, useful for smoke tests.')
    group.add_argument('--max-val-recordings', type=int, default=None,
                       help='Optional cap on the number of validation recordings to load, useful for smoke tests.')
    group.add_argument('--max-test-recordings', type=int, default=None,
                       help='Optional cap on the number of test recordings to load, useful for smoke tests.')
    group.add_argument('--max-train-segments', type=int, default=None,
                       help='Optional cap on the number of training segments after key generation, useful for smoke tests.')
    group.add_argument('--max-val-segments', type=int, default=None,
                       help='Optional cap on the number of validation segments after key generation, useful for smoke tests.')
    group.add_argument('--max-test-segments', type=int, default=None,
                       help='Optional cap on the number of test segments after key generation, useful for smoke tests.')


def add_training_args(parser):
    group = parser.add_argument_group('training arguments')
    group.add_argument('--batch-size', type=int, default=2048,
                       help='Training batch size.')
    group.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs.')
    group.add_argument('--lr', type=float, default=0.01,
                       help='Initial learning rate.')
    group.add_argument('--l2', type=float, default=0.01,
                       help='L2 regularization / optimizer weight decay.')
    group.add_argument('--dropout-rate', type=float, default=0.5,
                       help='Dropout rate used by neural network models.')
    group.add_argument('--seed', type=int, default=1,
                       help='Random seed.')
    group.add_argument('--warmup-epochs', type=int, default=0,
                       help='Number of initial epochs that add Gaussian noise to real EEG inputs before normal training.')
    group.add_argument('--random-label', action='store_true',
                       help='During warm-up epochs, replace labels with uniformly random 0/1 labels unpaired with inputs.')


def add_runtime_args(parser):
    group = parser.add_argument_group('runtime and workflow arguments')
    group.add_argument('--device', default=None,
                       help='Device for training/inference, e.g. cpu, cuda, cuda:0, cuda:1. Defaults to cuda if available.')
    group.add_argument('--load-generators', action=argparse.BooleanOptionalAction, default=True,
                       help='Load cached training and validation datasets (default: enabled). '
                            'Use --no-load-generators to regenerate from raw EDF files.')
    group.add_argument('--save-generators', action='store_true',
                       help='Save generated training and validation datasets.')
    group.add_argument('--skip-train', action='store_true',
                       help='Skip training and only run later stages.')
    group.add_argument('--skip-predict', action='store_true',
                       help='Skip prediction.')
    group.add_argument('--skip-evaluate', action='store_true',
                       help='Skip evaluation.')


def add_method_args(parser):
    SVM.add_cli_args(parser)
    XGB.add_cli_args(parser)
    STEEGFormer.add_cli_args(parser)
    BIOT.add_cli_args(parser)
    BENDR.add_cli_args(parser)
    CBraMod.add_cli_args(parser)
    Conformer.add_cli_args(parser)
    EEGPT.add_cli_args(parser)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train, predict, and evaluate seizure detection models on SeizeIT2.'
    )
    add_experiment_args(parser)
    add_data_args(parser)
    add_training_args(parser)
    add_runtime_args(parser)
    add_method_args(parser)
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    key_generator.random.seed(seed)


def apply_experiment_config(config, args):
    config.model = args.model
    config.dataset = args.dataset
    config.save_dir = args.save_dir
    os.makedirs(config.save_dir, exist_ok=True)


def apply_data_config(config, args):
    config.data_path = args.data_path
    config.fs = args.fs
    config.CH = args.channels
    config.cross_validation = args.cross_validation
    config.frame = args.frame
    config.stride = args.stride
    config.stride_s = args.stride_s
    config.boundary = args.boundary
    config.factor = args.factor
    config.sample_type = args.sample_type
    config.max_train_recordings = args.max_train_recordings
    config.max_val_recordings = args.max_val_recordings
    config.max_test_recordings = args.max_test_recordings
    config.max_train_segments = args.max_train_segments
    config.max_val_segments = args.max_val_segments
    config.max_test_segments = args.max_test_segments


def apply_training_config(config, args):
    config.batch_size = args.batch_size
    config.dropoutRate = args.dropout_rate
    config.nb_epochs = args.epochs
    config.l2 = args.l2
    config.lr = args.lr
    config.seed = args.seed
    config.warmup_epochs = args.warmup_epochs
    config.random_label = args.random_label


def apply_runtime_config(config, args):
    if args.device is not None:
        config.device = args.device


def apply_method_config(config, args):
    configure = METHOD_CONFIGURERS.get(args.model)
    if configure is not None:
        configure(config, args)


def build_config(args):
    config = Config()
    apply_experiment_config(config, args)
    apply_data_config(config, args)
    apply_training_config(config, args)
    apply_runtime_config(config, args)
    apply_method_config(config, args)
    return config


def main():
    args = parse_args()
    set_random_seed(args.seed)
    config = build_config(args)

    if not args.skip_train:
        main_func.train(config, args.load_generators, args.save_generators)

    if not args.skip_predict:
        print('Getting predictions on the test set...')
        main_func.predict(config)

    if not args.skip_evaluate:
        print('Getting evaluation metrics...')
        main_func.evaluate(config)


if __name__ == '__main__':
    main()
