import csv
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.eeg_utils import decay_schedule, validation_metrics, weighted_focal_loss

WARMUP_NOISE_STD = 1.0


def _get_device(config, force_cpu=False):
    if force_cpu:
        return torch.device('cpu')
    if hasattr(config, 'device'):
        return torch.device(config.device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _make_loader(dataset, batch_size, shuffle, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _to_device(x, device):
    if isinstance(x, dict):
        return {key: value.to(device) for key, value in x.items()}
    return x.to(device)


def _forward(model, x):
    output = model(x)
    if isinstance(output, dict):
        return output['M']
    return output


def _write_history_header(history_path):
    if not os.path.exists(history_path) or os.path.getsize(history_path) == 0:
        with open(history_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                'epoch', 'loss', 'accuracy', 'sens', 'spec', 'sens_ovlp',
                'fah_ovlp', 'fah_epoch', 'faRate_epoch', 'score', 'auc',
                'val_loss', 'val_accuracy', 'val_sens', 'val_spec', 'val_sens_ovlp',
                'val_fah_ovlp', 'val_fah_epoch', 'val_faRate_epoch', 'val_score', 'val_auc',
                'lr',
            ])


def _append_history(history_path, row):
    with open(history_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(row)


def _run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    desc=None,
    warmup_noise=False,
    warmup_random_label=False,
):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_targets = []
    start_time = time.time()
    progress = tqdm(
        loader,
        desc=desc or ('train' if train else 'val'),
        unit='batch',
        dynamic_ncols=True,
        leave=True,
    )

    for x, y in progress:
        x = _to_device(x, device)
        y = y.to(device)

        if train and warmup_noise:
            x = x + WARMUP_NOISE_STD * torch.randn_like(x)
            if warmup_random_label:
                y = torch.randint(0, 2, (y.size(0),), device=y.device, dtype=y.dtype)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = _forward(model, x)
            loss = weighted_focal_loss(logits, y)

            if train:
                loss.backward()
                optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        probs = torch.softmax(logits.detach(), dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())
        elapsed = max(time.time() - start_time, 1e-9)
        progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'samples': total_samples,
            'samples/s': f'{total_samples / elapsed:.1f}',
        })

    y_prob = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.uint8)
    y_pred = (y_prob >= 0.5).astype(np.uint8)
    avg_loss = total_loss / total_samples if total_samples else 0.0

    if train:
        metrics = {
            'accuracy': float(np.mean(y_pred == y_true)) if len(y_true) else 0.0,
            'sens': float('nan'),
            'spec': float('nan'),
            'sens_ovlp': float('nan'),
            'fah_ovlp': float('nan'),
            'fah_epoch': float('nan'),
            'faRate_epoch': float('nan'),
            'score': float('nan'),
            'auc': float('nan'),
        }
    else:
        print(f"Computing {desc or 'validation'} metrics...")
        metrics = validation_metrics(y_true, y_prob)

    return avg_loss, metrics


def _save_checkpoint(path, model, optimizer, epoch, val_score, config):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_score': val_score,
        'config': dict(config.__dict__),
    }, path)


def _load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def train_net(config, model, gen_train, gen_val, model_save_path):
    ''' Routine to train the model with the desired configurations.

        Args:
            config: configuration object containing all parameters
            model: PyTorch model object
            gen_train: a PyTorch dataset containing the training data
            gen_val: a PyTorch dataset containing the validation data
            model_save_path: path to the folder to save the models' weights
    '''

    name = config.get_name()
    device = _get_device(config)
    model = model.to(device)
    print(model)
    print('Using device:', device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.999),
        weight_decay=getattr(config, 'l2', 0),
    )

    callbacks_path = os.path.join(model_save_path, 'Callbacks')
    history_dir = os.path.join(model_save_path, 'History')
    weights_dir = os.path.join(model_save_path, 'Weights')
    os.makedirs(callbacks_path, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    train_loader = _make_loader(gen_train, config.batch_size, shuffle=True, drop_last=True)
    val_batch_size = getattr(gen_val, 'batch_size', config.batch_size)
    val_loader = _make_loader(gen_val, val_batch_size, shuffle=False, drop_last=False)
    print(
        f"Train samples: {len(gen_train)} | train batches/epoch: {len(train_loader)} | "
        f"Val samples: {len(gen_val)} | val batches/epoch: {len(val_loader)}"
    )

    history_path = os.path.join(history_dir, name + '.csv')
    _write_history_header(history_path)

    warmup_epochs = max(0, int(getattr(config, 'warmup_epochs', 0) or 0))
    warmup_random_label = bool(getattr(config, 'random_label', False))
    if warmup_epochs > 0:
        label_mode = 'random labels' if warmup_random_label else 'real labels from loader'
        print(
            f'Warm-up enabled: {warmup_epochs} epoch(s) adding N(0, {WARMUP_NOISE_STD}^2) noise '
            f'to real EEG with {label_mode}.'
        )
    elif warmup_random_label:
        print('Ignoring --random-label because --warmup-epochs is 0.')

    best_score = -np.inf
    best_checkpoint_path = None

    for epoch in range(1, config.nb_epochs + 1):
        epoch_start = time.time()
        in_warmup = epoch <= warmup_epochs
        train_desc = f'Epoch {epoch}/{config.nb_epochs} train'
        if in_warmup:
            train_desc += ' [warm-up]'
        train_loss, train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            desc=train_desc,
            warmup_noise=in_warmup,
            warmup_random_label=in_warmup and warmup_random_label,
        )
        val_loss, val_metrics = _run_epoch(
            model,
            val_loader,
            device,
            desc=f'Epoch {epoch}/{config.nb_epochs} val',
        )
        epoch_time = time.time() - epoch_start

        lr = optimizer.param_groups[0]['lr']
        checkpoint_path = os.path.join(callbacks_path, f'{name}_{epoch:02d}.pt')
        _save_checkpoint(checkpoint_path, model, optimizer, epoch, val_metrics['score'], config)

        if epoch > warmup_epochs and val_metrics['score'] > best_score:
            best_score = val_metrics['score']
            best_checkpoint_path = checkpoint_path

        _append_history(history_path, [
            epoch,
            train_loss,
            train_metrics['accuracy'],
            train_metrics['sens'],
            train_metrics['spec'],
            train_metrics['sens_ovlp'],
            train_metrics['fah_ovlp'],
            train_metrics['fah_epoch'],
            train_metrics['faRate_epoch'],
            train_metrics['score'],
            train_metrics['auc'],
            val_loss,
            val_metrics['accuracy'],
            val_metrics['sens'],
            val_metrics['spec'],
            val_metrics['sens_ovlp'],
            val_metrics['fah_ovlp'],
            val_metrics['fah_epoch'],
            val_metrics['faRate_epoch'],
            val_metrics['score'],
            val_metrics['auc'],
            lr,
        ])

        print(
            f"Epoch {epoch}/{config.nb_epochs} - "
            f"loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
            f"val_score: {val_metrics['score']:.4f} - lr: {lr:.6g} - "
            f"time: {epoch_time / 60:.2f} min"
        )

        new_lr = decay_schedule(epoch - 1, lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    if best_checkpoint_path is None and warmup_epochs < config.nb_epochs:
        best_checkpoint_path = os.path.join(callbacks_path, f'{name}_{config.nb_epochs:02d}.pt')

    if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
        checkpoint = _load_checkpoint(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_score': best_score,
            'config': dict(config.__dict__),
        }, os.path.join(weights_dir, name + '.pt'))

    print("Saved model to disk")


def predict_net(generator, model_weights_path, model, batch_size=None, device=None):
    ''' Routine to obtain predictions from the trained model with the desired configurations.

    Args:
        generator: a PyTorch dataset containing the data to predict
        model_weights_path: path to the model checkpoint
        model: PyTorch model object

    Returns:
        y_pred: array with the probability of seizure occurrences (0 to 1) of each consecutive
                window of the recording.
        y_true: analogous to y_pred, the array contains the label of each segment (0 or 1)
    '''

    device = torch.device(device) if device is not None else torch.device('cpu')
    model = model.to(device)
    checkpoint = _load_checkpoint(model_weights_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    loader = _make_loader(generator, batch_size or len(generator), shuffle=False, drop_last=False)
    preds = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = _to_device(x, device)
            logits = _forward(model, x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            preds.append(prob.cpu().numpy())
            labels.append(y.numpy())

    y_pred = np.concatenate(preds).astype('float32') if preds else np.array([], dtype='float32')
    y_true = np.concatenate(labels).astype('uint8') if labels else np.array([], dtype='uint8')
    return y_pred, y_true
