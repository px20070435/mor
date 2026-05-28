import pickle
import os

class Config():
    """ Class to create and store an experiment configuration object with the architecture hyper-parameters, input and sampling types.
    
    Args:
        data_path (str): path to data
        model (str): model architecture (you have 3: Chrononet, EEGnet, DeepConvNet)
        dataset (str): patients split  (check 'datasets' folder)

        fs (int): desired sampling frequency of the input data.
        CH (int): number of channels of the input data.
        frame (int): window size of input segments in seconds.
        stride (float): stride between segments (of background EEG) in seconds
        stride_s (float): stride between segments (of seizure EEG) in seconds
        boundary (float): proportion of seizure data in a window to consider the segment in the positive class
        batch_size (int): batch size for training model
        sample_type (str): sampling method (default is subsample, removes background EEG segments to match the number of seizure segments times the balancing factor)
        factor(int): balancing factor between number of segments in each class. The number of background segments is the number of seizure segments times the balancing factor.
        l2 (float): L2 regularization penalty
        lr (float): learning rate
        dropoutRate (float): layer's dropout rate
        nb_epochs (int): number of epochs to train model
        class_weights (dict): weight of each class for computing the loss function
        cross_validation (str): validation type (default is 'fixed' set of patients for training and validation)
        save_dir (str): save directory for intermediate and output files

    """

    def __init__(self, data_path=None, model='ChronoNet', dataset='SZ2', fs=None, CH=None, frame=2, stride=1, stride_s=0.5, boundary=0.5, batch_size=64, sample_type='subsample', factor=5, l2=0, lr=0.01, dropoutRate=0, nb_epochs=50, class_weights = {0:1, 1:1}, cross_validation='fixed', save_dir='savedir'):

        self.data_path = data_path
        self.model = model
        self.dataset = dataset
        self.save_dir = save_dir
        self.fs = fs
        self.CH = CH
        self.frame = frame
        self.stride = stride
        self.stride_s = stride_s
        self.boundary = boundary
        self.batch_size = batch_size
        self.sample_type = sample_type
        self.factor = factor
        self.cross_validation = cross_validation
        self.savedir = save_dir

        # models parameters
        self.data_format = 'channels_first'
        self.l2 = l2
        self.lr = lr
        self.dropoutRate = dropoutRate
        self.nb_epochs = nb_epochs
        self.class_weights = class_weights
        self.num_classes = 2
        self.train_mode = 'classification'
        self.seed = 1
        self.log_name = 'experiments.csv'
        self.max_train_recordings = None
        self.max_val_recordings = None
        self.max_test_recordings = None
        self.max_train_segments = None
        self.max_val_segments = None
        self.max_test_segments = None
        self.warmup_epochs = 0
        self.random_label = False

    def save_config(self, save_path):
        name = self.get_name()
        with open(os.path.join(save_path, name + '.cfg'), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)


    def load_config(self, config_path, config_name):
        if not os.path.exists(config_path):
            raise ValueError('Directory is empty or does not exist')

        with open(os.path.join(config_path, config_name), 'rb') as input:
            config = pickle.load(input)

        self.__dict__.update(config)

        
    def get_name(self):
        parts = [self.model, self.sample_type, 'factor' + str(self.factor)]
        if self.model == 'STEEGFormer':
            parts.extend([
                getattr(self, 'steegformer_variant', 'small'),
                'fs' + str(getattr(self, 'steegformer_target_fs', 128)),
            ])
            if getattr(self, 'steegformer_freeze_backbone', False):
                parts.append('frozen')
        elif self.model == 'BIOT':
            parts.append('fs' + str(getattr(self, 'biot_target_fs', 200)))
        elif self.model == 'BENDR':
            parts.append('fs' + str(getattr(self, 'bendr_target_fs', 256)))
        elif self.model == 'CBraMod':
            parts.append('fs' + str(getattr(self, 'cbramod_target_fs', 200)))
        elif self.model == 'EEGPT':
            parts.append('fs' + str(getattr(self, 'eegpt_target_fs', 256)))
        warmup_epochs = getattr(self, 'warmup_epochs', 0) or 0
        if warmup_epochs > 0:
            parts.append('warmup' + str(warmup_epochs))
            if getattr(self, 'random_label', False):
                parts.append('randlabel')
        return '_'.join(parts)
