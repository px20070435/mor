import pickle

from methods.SVM import extract_dataset, extract_generator_dataset, load_model, save_model


def add_cli_args(parser):
    group = parser.add_argument_group('XGB method arguments')
    group.add_argument('--xgb-n-estimators', type=int, default=300,
                       help='Number of boosting rounds for the XGB baseline.')
    group.add_argument('--xgb-max-depth', type=int, default=6,
                       help='Maximum tree depth for the XGB baseline.')
    group.add_argument('--xgb-learning-rate', type=float, default=0.05,
                       help='Learning rate for the XGB baseline.')
    group.add_argument('--xgb-subsample', type=float, default=0.8,
                       help='Row subsampling ratio for the XGB baseline.')
    group.add_argument('--xgb-colsample-bytree', type=float, default=0.8,
                       help='Feature subsampling ratio per tree.')
    group.add_argument('--xgb-reg-lambda', type=float, default=1.0,
                       help='L2 regularization term for the XGB baseline.')
    group.add_argument('--xgb-n-jobs', type=int, default=-1,
                       help='Number of CPU threads used by the XGB baseline.')
    group.add_argument('--xgb-tree-method', default='hist',
                       help='XGBoost tree construction method, e.g. hist, approx, auto.')


def apply_cli_args(config, args):
    config.xgb_n_estimators = args.xgb_n_estimators
    config.xgb_max_depth = args.xgb_max_depth
    config.xgb_learning_rate = args.xgb_learning_rate
    config.xgb_subsample = args.xgb_subsample
    config.xgb_colsample_bytree = args.xgb_colsample_bytree
    config.xgb_reg_lambda = args.xgb_reg_lambda
    config.xgb_n_jobs = args.xgb_n_jobs
    config.xgb_tree_method = args.xgb_tree_method


def net(config):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=getattr(config, 'xgb_n_estimators', 300),
        max_depth=getattr(config, 'xgb_max_depth', 6),
        learning_rate=getattr(config, 'xgb_learning_rate', 0.05),
        subsample=getattr(config, 'xgb_subsample', 0.8),
        colsample_bytree=getattr(config, 'xgb_colsample_bytree', 0.8),
        reg_lambda=getattr(config, 'xgb_reg_lambda', 1.0),
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=getattr(config, 'seed', 1),
        n_jobs=getattr(config, 'xgb_n_jobs', -1),
        tree_method=getattr(config, 'xgb_tree_method', 'hist'),
    )


__all__ = [
    'extract_dataset',
    'extract_generator_dataset',
    'add_cli_args',
    'apply_cli_args',
    'load_model',
    'net',
    'save_model',
]
