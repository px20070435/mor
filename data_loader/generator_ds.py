import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data_loader.data import Data
from utils.eeg_utils import apply_preprocess_eeg


class _BaseEEGDataset(Dataset):
    """Preloaded EEG segment dataset used by the PyTorch training and prediction loops."""

    def __init__(self, config, segments):
        self.config = config
        self.data_segs = np.empty(shape=[len(segments), int(config.frame * config.fs), config.CH], dtype=np.float32)
        self.labels = np.empty(shape=[len(segments)], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_segs[index]
        if self.config.model in ('DeepConvNet', 'EEGnet'):
            x = x.T[np.newaxis, :, :]
        else:
            x = x.T
        return torch.from_numpy(x.astype(np.float32, copy=False)), torch.tensor(self.labels[index], dtype=torch.long)

    def _load_recording(self, rec):
        rec_data = Data.loadData(self.config.data_path, rec, modalities=['eeg'])
        return apply_preprocess_eeg(self.config, rec_data)

    def _store_segment(self, index, rec_data, start_seg, stop_seg):
        if stop_seg > len(rec_data[0]):
            self.data_segs[index, :, :] = 0
        else:
            self.data_segs[index, :, 0] = rec_data[0][start_seg:stop_seg]
            self.data_segs[index, :, 1] = rec_data[1][start_seg:stop_seg]


class SequentialGenerator(_BaseEEGDataset):
    ''' Class where a sequential PyTorch dataset is built (the data segments are continuous and aligned in time).

    Args:
        config (cls): config object with the experiment parameters
        recs (list[list[str]]): list of recordings in the format [sub-xxx, run-xx]
        segments: list of keys (each key is a list [1x4] containing the recording index in the rec list,
                  the start and stop of the segment in seconds and the label of the segment)
        batch_size: retained for compatibility with cached generator objects
        shuffle: retained for compatibility; DataLoader owns shuffling in the PyTorch pipeline
    '''

    def __init__(self, config, recs, segments, batch_size=32, shuffle=False, verbose=True):
        super().__init__(config, segments)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose

        pbar = tqdm(total=len(segments), disable=not self.verbose)

        count = 0
        prev_rec = int(segments[0][0])

        rec_data = self._load_recording(recs[prev_rec])

        for s in segments:
            curr_rec = int(s[0])

            if curr_rec != prev_rec:
                rec_data = self._load_recording(recs[curr_rec])
                prev_rec = curr_rec

            start_seg = int(s[1] * config.fs)
            stop_seg = int(s[2] * config.fs)

            self._store_segment(count, rec_data, start_seg, stop_seg)
            self.labels[count] = int(s[3])

            count += 1
            pbar.update(1)

        pbar.close()


class SegmentedGenerator(_BaseEEGDataset):
    ''' Class where the segmented PyTorch dataset is built for subsampled segments from multiple recordings.

    Args:
        config (cls): config object with the experiment parameters
        recs (list[list[str]]): list of recordings in the format [sub-xxx, run-xx]
        segments: list of keys (each key is a list [1x4] containing the recording index in the rec list,
                  the start and stop of the segment in seconds and the label of the segment)
        batch_size: retained for compatibility with cached generator objects
        shuffle: retained for compatibility; DataLoader owns shuffling in the PyTorch pipeline
    '''

    def __init__(self, config, recs, segments, batch_size=32, shuffle=True, verbose=True):
        super().__init__(config, segments)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose

        segs_to_load = list(segments)

        pbar = tqdm(total=len(segs_to_load), disable=not self.verbose)
        count = 0

        while segs_to_load:
            curr_rec = int(segs_to_load[0][0])
            comm_recs = [i for i, x in enumerate(segs_to_load) if x[0] == curr_rec]

            rec_data = self._load_recording(recs[curr_rec])

            for r in comm_recs:
                start_seg = int(segs_to_load[r][1] * config.fs)
                stop_seg = int(segs_to_load[r][2] * config.fs)

                self._store_segment(count, rec_data, start_seg, stop_seg)
                self.labels[count] = int(segs_to_load[r][3])

                count += 1
                pbar.update(1)

            segs_to_load = [s for i, s in enumerate(segs_to_load) if i not in comm_recs]

        pbar.close()
