"""
Utilities for downloading and unpacking the CIFAR-10 dataset, originally published
by Krizhevsky et al. and hosted here: https://www.cs.toronto.edu/~kriz/cifar.html
"""

import glob
import numpy as np
import os
import pickle
import sys
import tarfile
import time
from six.moves import urllib

def _print(s):
    print(s)
    sys.stdout.flush()

def unpickle(file):
    fo = open(file, 'rb')
    if (sys.version_info >= (3, 0)):
        import pickle
        d = pickle.load(fo, encoding='latin1')
    else:
        import cPickle
        d = cPickle.load(fo)
    fo.close()
    return {'x': d['data'].reshape((10000,3,32,32)), 'y': np.array(d['labels']).astype(np.uint8)}


'''
Loads the batch of transitions and creates transitions objects from them.
(s, a, r, s')
'''
def load(path, transitions_filenumber=-1, transitions_filename="transitions",
         subset='train', extend_to=1, action=None):
    '''
    It is used to compute the density of state-action pairs.
    rho(s, a) = rho_a(s) * marginal(a)
    Where rho(s, a) is the density of the pair (s, a)
    And rho_action is the density of state s estimated from all the states where action a was taken.
    See Bellemare's Unifying count-based Exploration and Intrinsic Motivation for more details
    '''
    if transitions_filenumber == -1:
      transitions_filenumber = max(map(lambda x: x[:-4].split("_")[-2], glob.glob(os.path.join(path, transitions_filename+'*'))))
    if transitions_filenumber == -1:
      raise ValueError("The transitions file was not found")
    filename = get_filename(path, transitions_filename, transitions_filenumber)
    with open(os.path.join(path, filename+"_rest.pkl"), "rb") as f:
      downsampled, actions, rewards, terms = pickle.load(f)
    # Add a bit of black around downsampled states so their size is a multiple of 4.
    downsampled = np.concatenate((downsampled, np.zeros((downsampled.shape[0], 42, 2))), axis=2)
    downsampled = np.concatenate((downsampled, np.zeros((downsampled.shape[0], 2, 44))), axis=1)
    if subset == 'all':
        new_actions = action + np.zeros(actions.shape, dtype=np.int32) if action is not None else actions
        extra = int(np.ceil(downsampled.shape[0]/extend_to)*extend_to - downsampled.shape[0])
        downsampled = np.concatenate((downsampled, np.zeros((extra, 44, 44))), axis=0)
        new_actions = np.concatenate((new_actions, np.zeros((extra), dtype=np.int32)), axis=0)
    states = downsampled.reshape(downsampled.shape+(1,))
    dataset_size = states.shape[0]
    train_set_size = int(dataset_size / extend_to) * extend_to
    test_set_size = dataset_size - train_set_size
    if subset == 'train':
        _print("The train set size is {}".format(train_set_size))
        return states[:train_set_size], actions[:train_set_size], actions[:train_set_size]
    elif subset == 'test':
        _print("The test set size is {}".format(test_set_size))
        return states[train_set_size:], actions[train_set_size:], actions[train_set_size:]
    elif subset == 'all':
        _print("The data set size is {}".format(dataset_size))
        return states[:dataset_size], new_actions[:dataset_size], actions[:dataset_size]
    else:
        raise ValueError("The subset has to be train or test")

def get_filename(path, transitions_filename, transitions_filenumber):
    full_name = glob.glob(os.path.join(path, transitions_filename+'*_'+str(transitions_filenumber)+'*'))[0]
    full_name = full_name.split('_rest')[0]
    full_name = full_name.split('_states')[0]
    return full_name.split("/")[-1].split("\\")[-1]

class DataLoader(object):
    """ an object that generates batches of CIFAR-10 data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False, action=None, filename='transitions'):
        """
        - data_dir is location where to store files
        - subset is train|test
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        # load CIFAR-10 training data to RAM
        _print("Loading data")
        start_time = time.time()
        self.data, self.labels, self.original_labels = load(data_dir, subset=subset, extend_to=batch_size, action=action, transitions_filename=filename)
        _print('Data loaded in {:.1f} seconds'.format(time.time() - start_time))
        # self.data = np.transpose(self.data, (0,2,3,1)) # (N,3,32,32) -> (N,32,32,3)

        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_obs(self):
        return self.data.shape[0]

    def get_num_labels(self):
        return np.amax(self.original_labels) + 1

    def get_stat_labels(self):
        return np.unique(self.original_labels, return_counts=True)

    def reset(self):
        self.p = 0

    def truncate(self, num):
        self.data, self.labels, self.original_labels = self.data[:num], self.labels[:num], self.original_labels[:num]

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)
