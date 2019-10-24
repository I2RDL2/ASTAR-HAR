
### two agumentation of audio data is needed
### 1. horizontal translation
### 2. add bg noise with different level
###     This noise can be generated in advance as single category and used 
###     as augmentation later

import os
import numpy as np
from .utils import random_balanced_partitions, random_partitions

class HAR_SP:
    DATA_PATH = os.path.join('data', 'har', 'HAR_DATA_lok_sp_16.npz')
    VALIDATION_SET_RATIO = 0.1  # 10% of the training set will be used as validation set
    VALIDATION_SET_SIZE = 5000
    UNLABELED = -1  # We will be using -1 for unlabeled
    input_dim = (17,16,9)
    label_dim = ()
    num_train = 0

    # n_labeled for number of labled data, all for using all labels
    # more_data stands for additional data without labels used for training
    # besides the original 4500 training data
    def __init__(self, data_seed=0, n_labeled='all', test_phase=False, 
        f_sup = False,dataset_detail=None):

        random = np.random.RandomState(seed=data_seed)
        self._load()
        if test_phase:
            self.evaluation, self.training = self._test_and_training()
        else:
            self.evaluation, self.training = self._validation_and_training(random)

        if n_labeled != 'all':
            n_labeled = int(n_labeled)
            self.training = self._unlabel(self.training, n_labeled, random)

    def _load(self):
        file_data = np.load(self.DATA_PATH)
        NUM_TRAIN = len(file_data['train_x'])
        self.num_train=NUM_TRAIN
        NUM_TEST = len(file_data['test_x'])
        self._train_data = self._data_array(NUM_TRAIN, file_data['train_x'], file_data['train_y'])
        self._test_data = self._data_array(NUM_TEST, file_data['test_x'], file_data['test_y'])

    def _data_array(self, expected_n, x_data, y_data):
        array = np.zeros(expected_n, dtype=[
            ('x', np.float32, self.input_dim),
            ('y', np.int32, self.label_dim)  
        ])
        array['x'] = x_data
        array['y'] = y_data
        return array

    def _validation_and_training(self, random):
        return random_partitions(self._train_data, self.VALIDATION_SET_SIZE, random)

    def _test_and_training(self):
        return self._test_data, self._train_data

    def _unlabel(self, data, n_labeled, random):
        labeled, unlabeled = random_balanced_partitions(
            data, n_labeled, labels=data['y'], random=random)
        unlabeled['y'] = self.UNLABELED
        new_labeled = labeled.copy()
        new_labeled['y']=-1
        self.num_train +=len(new_labeled)
        return np.concatenate([labeled, unlabeled,new_labeled])

    def __getitem__(self, key):
        return getattr(self,key)
