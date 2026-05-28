from sklearn.metrics import roc_auc_score
import numpy as np
from scipy import signal
import torch
import torch.nn.functional as F


def set_gpu():
    """
    Reports the CUDA devices available to PyTorch.
    """
    if torch.cuda.is_available():
        print(torch.cuda.device_count(), 'CUDA device(s) available')
    else:
        print('No CUDA device available; using CPU')


def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """
    :param logits: Raw model outputs with shape [batch, classes]
    :param targets: Class indices with shape [batch]
    :return: Output tensor.
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    alpha_t = torch.where(
        targets == 1,
        torch.full_like(p_t, alpha),
        torch.full_like(p_t, 1 - alpha),
    )
    return (-(alpha_t * (1 - p_t).pow(gamma) * log_p_t)).mean()


def weighted_focal_loss(logits, targets, gamma=2.0):
    """
    Batch-adaptive weighted focal loss for two-class logits.

    :param logits: Raw model outputs with shape [batch, 2]
    :param targets: Class indices with shape [batch]
    :return: Output tensor.
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    p = probs[:, 1].clamp_min(torch.finfo(probs.dtype).eps)
    q = probs[:, 0].clamp_min(torch.finfo(probs.dtype).eps)
    log_p = log_probs[:, 1]
    log_q = log_probs[:, 0]

    pos_count = (targets == 1).sum().clamp_min(1)
    neg_count = (targets == 0).sum()
    pos_weight = neg_count.to(logits.dtype) / pos_count.to(logits.dtype)

    pos_loss = -(q ** gamma) * log_p * pos_weight
    neg_loss = -(p ** gamma) * log_q
    loss = torch.where(targets == 1, pos_loss, neg_loss)
    return loss.mean()


def weighted_binary_crossentropy(zero_weight, one_weight):

    def _weighted_binary_crossentropy(logits, targets):
        weights = torch.where(
            targets == 1,
            torch.as_tensor(one_weight, dtype=logits.dtype, device=logits.device),
            torch.as_tensor(zero_weight, dtype=logits.dtype, device=logits.device),
        )
        loss = F.cross_entropy(logits, targets, reduction='none')
        return (weights * loss).mean()

    return _weighted_binary_crossentropy


def weighted_binary_crossentropy_adapt(logits, targets):
    pos_count = (targets == 1).sum().clamp_min(1)
    neg_count = (targets == 0).sum()
    one_wt = neg_count.to(logits.dtype) / pos_count.to(logits.dtype)
    weights = torch.where(targets == 1, one_wt, torch.ones_like(one_wt))
    loss = F.cross_entropy(logits, targets, reduction='none')
    return (weights * loss).mean()


def decay_schedule(epoch, lr):
    if lr > 1e-5:
        if (epoch + 1) % 10 == 0:
            lr = lr / 2
        
    return lr


#######################################
############### metrics ###############

def validation_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.uint8)

    TP_epoch, FP_epoch, TN_epoch, FN_epoch = perf_measure_epoch(y_true, y_pred)
    TP_ovlp, FP_ovlp, FN_ovlp = perf_measure_ovlp(y_true, y_pred, fs=1)

    sens_epoch = TP_epoch / (TP_epoch + FN_epoch) if TP_epoch + FN_epoch else 0.0
    spec_epoch = TN_epoch / (TN_epoch + FP_epoch) if TN_epoch + FP_epoch else 0.0
    sens_overlap = TP_ovlp / (TP_ovlp + FN_ovlp) if TP_ovlp + FN_ovlp else 0.0
    fah_overlap = FP_ovlp * 3600 / len(y_true) if len(y_true) else 0.0
    fah_ep = FP_epoch * 3600 / len(y_true) if len(y_true) else 0.0
    fa_rate = FP_epoch / len(y_true) if len(y_true) else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    return {
        'accuracy': float(np.mean(y_pred == y_true)) if len(y_true) else 0.0,
        'sens': float(sens_epoch),
        'spec': float(spec_epoch),
        'sens_ovlp': float(sens_overlap),
        'fah_ovlp': float(fah_overlap),
        'fah_epoch': float(fah_ep),
        'faRate_epoch': float(fa_rate),
        'score': float(sens_overlap * 100 - 0.4 * fa_rate),
        'auc': float(auc),
    }


#### Pre-process EEG data

def apply_preprocess_eeg(config, rec):

    idx_focal = [i for i, c in enumerate(rec.channels) if c == 'BTEleft SD']
    if not idx_focal:
        idx_focal = [i for i, c in enumerate(rec.channels) if c == 'BTEright SD']
    idx_cross = [i for i, c in enumerate(rec.channels) if c == 'CROSStop SD']
    if not idx_cross:
        idx_cross = [i for i, c in enumerate(rec.channels) if c == 'BTEright SD']

    ch_focal, _ = pre_process_ch(rec.data[idx_focal[0]], rec.fs[idx_focal[0]], config.fs)
    ch_cross, _ = pre_process_ch(rec.data[idx_cross[0]], rec.fs[idx_cross[0]], config.fs)
        
    # ch_focal = (ch_focal - np.mean(ch_focal))/np.std(ch_focal)
    # ch_cross = (ch_cross - np.mean(ch_cross))/np.std(ch_cross)

    return [ch_focal, ch_cross]


def pre_process_ch(ch_data, fs_data, fs_resamp):

    if fs_resamp != fs_data:
        ch_data = signal.resample(ch_data, int(fs_resamp*len(ch_data)/fs_data))
    
    b, a = signal.butter(4, 0.5/(fs_resamp/2), 'high')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, 60/(fs_resamp/2), 'low')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, [49.5/(fs_resamp/2), 50.5/(fs_resamp/2)], 'bandstop')
    ch_data = signal.filtfilt(b, a, ch_data)

    return ch_data, fs_resamp


#### EVENT & MASK MANIPULATION ###

def eventList2Mask(events, totalLen, fs):
    """Convert list of events to mask.
    
    Returns a logical array of length totalLen.
    All event epochs are set to True
    
    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        totalLen: length of array to return in samples
        fs: sampling frequency of the data in Hertz
    Return:
        mask: logical array set to True during event epochs and False the rest
              if the time.
    """
    mask = np.zeros((totalLen,))
    for event in events:
        for i in range(min(int(event[0]*fs), totalLen), min(int(event[1]*fs), totalLen)):
            mask[i] = 1
    return mask


def mask2eventList(mask, fs):
    """Convert mask to list of events.
        
    Args:
        mask: logical array set to True during event epochs and False the rest
          if the time.
        fs: sampling frequency of the data in Hertz
    Return:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
    """
    events = list()
    tmp = []
    start_i = np.where(np.diff(np.array(mask, dtype=int)) == 1)[0]
    end_i = np.where(np.diff(np.array(mask, dtype=int)) == -1)[0]
    
    if len(start_i) == 0 and mask[0]:
        events.append([0, (len(mask)-1)/fs])
    else:
        # Edge effect
        if mask[0]:
            events.append([0, (end_i[0]+1)/fs])
            end_i = np.delete(end_i, 0)
        # Edge effect
        if mask[-1]:
            if len(start_i):
                tmp = [[(start_i[-1]+1)/fs, (len(mask))/fs]]
                start_i = np.delete(start_i, len(start_i)-1)
        for i in range(len(start_i)):
            events.append([(start_i[i]+1)/fs, (end_i[i]+1)/fs])
        events += tmp
    return events


def merge_events(events, distance):
    """ Merge events.
    
    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        distance: maximum distance (in seconds) between events to be merged
    Return:
        events: list of events (after merging) times in seconds.
    """
    i = 1
    tot_len = len(events)
    while i < tot_len:
        if events[i][0] - events[i-1][1] < distance:
            events[i-1][1] = events[i][1]
            events.pop(i)
            tot_len -= 1
        else:
            i += 1
    return events


def get_events(events, margin):
    ''' Converts the unprocessed events to the post-processed events based on physiological constrains:
    - seizure alarm events distanced by 0.2*margin (in seconds) are merged together
    - only events with a duration longer than margin*0.8 are kept
    (for more info, check: K. Vandecasteele et al., “Visual seizure annotation and automated seizure detection using
    behind-the-ear elec- troencephalographic channels,” Epilepsia, vol. 61, no. 4, pp. 766–775, 2020.)

    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        margin: float, the desired margin in seconds

    Returns:
        ev_list: list of events times in seconds after merging and discarding short events.
    '''
    events_merge = merge_events(events, 0.2*margin)
    ev_list = []
    for i in range(len(events_merge)):
        if events_merge[i][1] - events_merge[i][0] >= margin*0.8:
            ev_list.append(events_merge[i])

    return ev_list



def post_processing(y_pred, fs, th, margin):
    ''' Post process the predictions given by the model based on physiological constraints: a seizure is
    not shorter than 10 seconds and events separated by 2 seconds are merged together.

    Args:
        y_pred: array with the seizure classification probabilties (of each segment)
        fs: sampling frequency of the y_pred array (1/window length - in this challenge fs = 1/2)
        th: threshold value for seizure probability (float between 0 and 1)
        margin: float, the desired margin in seconds (check get_events)
    
    Returns:
        pred: array with the processed classified labels by the model
    '''
    pred = (y_pred > th)
    events = mask2eventList(pred, fs)
    events = get_events(events, margin)
    pred = eventList2Mask(events, len(y_pred), fs)

    return pred


def getOverlap(a, b):
    ''' If > 0, the two intervals overlap.
    a = [start_a, end_a]; b = [start_b, end_b]
    '''
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def perf_measure_epoch(y_true, y_pred):
    ''' Calculate the performance metrics based on the EPOCH method.
    
    Args:
        y_true: array with the ground-truth labels of the segments
        y_pred: array with the predicted labels of the segments

    Returns:
        TP: true positives
        FP: false positives
        TN: true negatives
        FN: false negatives
    '''

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_true[i] == y_pred[i] == 1:
           TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
           FP += 1
        if y_true[i] == y_pred[i] == 0:
           TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
           FN += 1

    return TP, FP, TN, FN


def perf_measure_ovlp(y_true, y_pred, fs):
    ''' Calculate the performance metrics based on the any-overlap method.
    
    Args:
        y_true: array with the ground-truth labels of the segments
        y_pred: array with the predicted labels of the segments
        fs: sampling frequency of the predicted and ground-truth label arrays
            (in this challenge, fs = 1/2)

    Returns:
        TP: true positives
        FP: false positives
        FN: false negatives
    '''
    true_events = mask2eventList(y_true, fs)
    pred_events = mask2eventList(y_pred, fs)

    TP = 0
    FP = 0
    FN = 0

    for pr in pred_events:
        found = False
        for tr in true_events:
            if getOverlap(pr, tr) > 0:
                TP += 1
                found = True
        if not found:
            FP += 1
    for tr in true_events:
        found = False
        for pr in pred_events:
            if getOverlap(tr, pr) > 0:
                found = True
        if not found:
            FN += 1

    return TP, FP, FN


def get_metrics_scoring(y_pred, y_true, fs, th):
    ''' Get the score for the challenge.

    Args:
        pred_file: path to the prediction file containing the objects 'filenames',
                   'predictions' and 'labels' (as returned by 'predict_net' function)
    
    Returns:
        score: the score of the challenge
        sens_ovlp: sensitivity calculated with the any-overlap method
        FA_epoch: false alarm rate (false alarms per hour) calculated with the EPOCH method
    '''

    total_N = len(y_pred)*(1/fs)
    total_seiz = np.sum(y_true)

    # Post process predictions (merge predicted events separated by 2 second and discard events smaller than 8 seconds)
    y_pred = post_processing(y_pred, fs=fs, th=th, margin=10)

    TP_epoch, FP_epoch, TN_epoch, FN_epoch = perf_measure_epoch(y_true, y_pred)

    TP_ovlp, FP_ovlp, FN_ovlp = perf_measure_ovlp(y_true, y_pred, fs=1/2)

    if total_seiz == 0:
        sens_ovlp = float("nan")
        prec_ovlp = float("nan")
        f1_ovlp = float("nan")
    else:
        sens_ovlp = TP_ovlp/(TP_ovlp + FN_ovlp)
        if TP_ovlp == 0 and FP_ovlp == 0:
            prec_ovlp = float("nan")
            f1_ovlp = float("nan")
        else:
            prec_ovlp = TP_ovlp/(TP_ovlp + FP_ovlp)
            if prec_ovlp+sens_ovlp == 0:
                f1_ovlp = float("nan")
            else:
                f1_ovlp = (2*prec_ovlp*sens_ovlp)/(prec_ovlp+sens_ovlp)
    
    FA_ovlp = FP_ovlp*3600/total_N
    FA_epoch = FP_epoch*3600/total_N

    if total_seiz == 0:
        sens_epoch = float("nan")
        prec_epoch = float("nan")
        f1_epoch = float("nan")
    else:
        sens_epoch = TP_epoch/(TP_epoch + FN_epoch)
        if TP_ovlp == 0 and FP_ovlp == 0:
            prec_epoch = float("nan")
            f1_epoch = float("nan")
        else:
            prec_epoch = TP_epoch/(TP_epoch + FP_epoch)
            if prec_epoch+sens_epoch == 0:
                f1_epoch = float("nan")
            else:
                f1_epoch = (2*prec_epoch*sens_epoch)/(prec_epoch+sens_epoch)

    spec_epoch = TN_epoch/(TN_epoch + FP_epoch)

    return sens_ovlp, prec_ovlp, FA_ovlp, f1_ovlp, sens_epoch, spec_epoch, prec_epoch, FA_epoch, f1_epoch
