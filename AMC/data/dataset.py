import os
import h5py
import json
import numpy as np
import random
import torch
import torch.utils.data as data
from data.transform import AMCTransform
from runner.utils import get_config
from numpy import argwhere

random.seed(100)


class AMCTrainDataset(data.Dataset):
    def __init__(self, root_path, mode=None, robust=False, snr_range=None):
        super(AMCTrainDataset, self).__init__()

        self.root_path = root_path
        self.mode = mode
        self.snr_range = snr_range
        self.transforms = AMCTransform()
        self.robust = robust
        self.config = get_config('config.yaml')

        self.data = h5py.File(os.path.join(self.root_path, "GOLD_XYZ_OSC.0001_1024.hdf5"), 'r')
        self.class_labels = json.load(open(os.path.join(self.root_path, "classes-fixed.json"), 'r'))

        self.iq = self.data['X']
        self.onehot = self.data['Y']
        self.snr = np.squeeze(self.data['Z'])
        self.num_modulation = 24
        self.num_sample = int(4096 * self.config['train_proportion'])  # sample per modulation-snr

        # Sampling data in snr boundary
        if self.snr_range is not None:
            snr_mask = (self.snr_range[0] <= self.snr) & (self.snr <= self.snr_range[1])
            self.iq = self.iq[snr_mask]
            self.onehot = self.onehot[snr_mask]
            self.snr = self.snr[snr_mask]

        # Sampling data which are easy modulation
        if self.mode == 'easy':
            mod_mask = np.array([int(argwhere(self.onehot[i] == 1)) in self.config['easy_class_indice'] for i in range(len(self.onehot))])
            self.iq = self.iq[mod_mask]
            self.onehot = self.onehot[mod_mask]
            self.snr = self.snr[mod_mask]
            self.num_modulation = len(self.config['easy_class_indice'])

        # Sampling train data
        # each modulation-snr has 4096 I/Q samples
        sampling_mask = []
        for _ in range(self.num_modulation * len(np.unique(self.snr))):
            sampling_mask.extend([True for _ in range(self.num_sample)])
            sampling_mask.extend([False for _ in range(4096-self.num_sample)])
        sampling_mask = np.array(sampling_mask)

        self.iq = self.iq[sampling_mask]
        self.onehot = self.onehot[sampling_mask]
        self.snr = self.snr[sampling_mask]

    def __len__(self):
        return self.iq.shape[0]

    def __getitem__(self, item):
        label = int(argwhere(self.onehot[item] == 1))
        x = self.iq[item].transpose()

        if self.robust is True:
            revers = np.flip(x.copy(), axis=1)
            x = np.concatenate((x, revers), axis=0)

            sample = {"data": self.transforms(x), "label": label, "snr": self.snr[item]}  # self.transforms(x)
        else:
            sample = {"data": x, "label": label, "snr": self.snr[item]}

        return sample


class AMCTestDataset(data.Dataset):
    def __init__(self, root_path, mode=None, robust=False, snr_range=None):
        super(AMCTestDataset, self).__init__()

        self.root_path = root_path
        self.mode = mode
        self.snr_range = snr_range
        self.transforms = AMCTransform()
        self.robust = robust
        self.config = get_config('config.yaml')

        self.data = h5py.File(os.path.join(self.root_path, "GOLD_XYZ_OSC.0001_1024.hdf5"), 'r')
        self.class_labels = json.load(open(os.path.join(self.root_path, "classes-fixed.json"), 'r'))

        self.iq = self.data['X']
        self.onehot = self.data['Y']
        self.snr = np.squeeze(self.data['Z'])
        self.num_modulation = 24
        self.num_sample = 4096 - int(4096 * self.config['train_proportion'])  # sample per modulation-snr

        # Sampling data in snr boundary
        if self.snr_range is not None:
            snr_mask = (self.snr_range[0] <= self.snr) & (self.snr <= self.snr_range[1])
            self.iq = self.iq[snr_mask]
            self.onehot = self.onehot[snr_mask]
            self.snr = self.snr[snr_mask]

        # Sampling data which are easy modulation
        if self.mode == 'easy':
            mod_mask = np.array([int(argwhere(self.onehot[i] == 1)) in self.config['easy_class_indice'] for i in
                                 range(len(self.onehot))])
            self.iq = self.iq[mod_mask]
            self.onehot = self.onehot[mod_mask]
            self.snr = self.snr[mod_mask]
            self.num_modulation = len(self.config['easy_class_indice'])

        # Sampling train data
        # each modulation-snr has 4096 I/Q samples
        sampling_mask = []
        for _ in range(self.num_modulation * len(np.unique(self.snr))):
            sampling_mask.extend([False for _ in range(4096 - self.num_sample)])
            sampling_mask.extend([True for _ in range(self.num_sample)])
        sampling_mask = np.array(sampling_mask)

        self.iq = self.iq[sampling_mask]
        self.onehot = self.onehot[sampling_mask]
        self.snr = self.snr[sampling_mask]

    def __len__(self):
        return self.iq.shape[0]

    def __getitem__(self, item):
        label = int(argwhere(self.onehot[item] == 1))
        x = self.iq[item].transpose()

        if self.robust is True:
            revers = np.flip(x.copy(), axis=1)
            x = np.concatenate((x, revers), axis=0)

            sample = {"data": self.transforms(x), "label": label, "snr": self.snr[item]}  # self.transforms(x)
        else:
            sample = {"data": x, "label": label, "snr": self.snr[item]}

        return sample


class FewShotDataset(data.Dataset):
    def __init__(self, root_path, num_support, num_query, mode='train', robust=False, snr_range=None):
        self.root_path = root_path
        self.robust = robust
        self.snr_range = snr_range
        self.mode = mode
        self.config = get_config('config.yaml')

        self.data = h5py.File(os.path.join(self.root_path, "GOLD_XYZ_OSC.0001_1024.hdf5"), 'r')
        self.class_labels = json.load(open(os.path.join(self.root_path, "classes-fixed.json"), 'r'))

        self.iq = self.data['X']
        self.onehot = self.data['Y']
        self.snr = np.squeeze(self.data['Z'])

        # Sampling data in snr boundary
        if self.snr_range is not None:
            snr_mask = (self.snr_range[0] <= self.snr) & (self.snr <= self.snr_range[1])
            self.iq = self.iq[snr_mask]
            self.onehot = self.onehot[snr_mask]
            self.snr = self.snr[snr_mask]

        # Sampling class
        if mode == 'train':
            mod_mask = np.array([int(argwhere(self.onehot[i] == 1)) in self.config['easy_class_indice'] for i in
                                 range(len(self.onehot))])
            self.num_modulation = len(self.config['easy_class_indice'])
        elif mode == 'test':
            mod_mask = np.array([int(argwhere(self.onehot[i] == 1)) in self.config['difficult_class_indice'] for i in
                                 range(len(self.onehot))])
            self.num_modulation = len(self.config['difficult_class_indice'])
        else:
            print('Mode argument error!')
            exit()

        self.iq = self.iq[mod_mask]
        self.onehot = self.onehot[mod_mask]
        self.snr = self.snr[mod_mask]

        # Extract class labels
        self.label_list = [int(argwhere(self.onehot[i] == 1)) for i in range(len(self.snr))]
        self.labels = np.unique(self.label_list)

        # Extract indice of each labels
        self.label_indices = {label: [i for i, x in enumerate(self.label_list) if x == label] for label in self.labels}

        # few-shot variables
        self.num_support = num_support
        self.num_query = num_query
        self.num_episode = len(self.snr) // ((self.num_support + self.num_query) * len(self.labels))

    def __len__(self):
        return self.num_episode

    def __getitem__(self, idx):
        # idx means index of episode
        sample = dict()

        if self.robust is True:
            for label in self.labels:
                sample[label] = dict()
                label_indices = self.label_indices[label]

                # support set
                support_indices = random.sample(label_indices, self.num_support)
                support_set = [np.concatenate((self.iq[i].transpose(), np.flip(self.iq[i].transpose(), axis=1)), axis=0)
                               for i in support_indices]
                sample[label]['support'] = support_set

                # query set
                query_indices = list(set(label_indices) - set(support_indices))
                query_indices = random.sample(query_indices, self.num_query)
                query_set = [np.concatenate((self.iq[i].transpose(), np.flip(self.iq[i].transpose(), axis=1)), axis=0)
                             for i in query_indices]
                sample[label]['query'] = query_set

        else:
            for label in self.labels:
                sample[label] = dict()
                label_indices = self.label_indices[label]

                # support set
                support_indices = random.sample(label_indices, self.num_support)
                support_set = [self.iq[i].transpose() for i in support_indices]
                sample[label]['support'] = support_set

                # query set
                query_indices = list(set(label_indices) - set(support_indices))
                query_indices = random.sample(query_indices, self.num_query)
                query_set = [self.iq[i].transpose() for i in query_indices]
                sample[label]['query'] = query_set

        return sample


class FewShotDatasetForOnce(data.Dataset):
    def __init__(self, root_path, num_support, num_query, robust=False, snr_range=None):
        self.root_path = root_path
        self.robust = robust
        self.snr_range = snr_range
        self.config = get_config('config.yaml')

        self.data = h5py.File(os.path.join(self.root_path, "GOLD_XYZ_OSC.0001_1024.hdf5"), 'r')
        self.class_labels = json.load(open(os.path.join(self.root_path, "classes-fixed.json"), 'r'))

        self.iq = self.data['X']
        self.onehot = self.data['Y']
        self.snr = np.squeeze(self.data['Z'])

        # Sampling data in snr boundary
        if self.snr_range is not None:
            snr_mask = (self.snr_range[0] <= self.snr) & (self.snr <= self.snr_range[1])
            self.iq = self.iq[snr_mask]
            self.onehot = self.onehot[snr_mask]
            self.snr = self.snr[snr_mask]

        # Sampling class
        mod_mask = np.array([int(argwhere(self.onehot[i] == 1)) in self.config['difficult_class_indice'] for i in range(len(self.onehot))])
        self.num_modulation = len(self.config['difficult_class_indice'])

        self.iq = self.iq[mod_mask]
        self.onehot = self.onehot[mod_mask]
        self.snr = self.snr[mod_mask]

        # Extract class labels
        self.label_list = [int(argwhere(self.onehot[i] == 1)) for i in range(len(self.snr))]
        self.labels = np.unique(self.label_list)

        # Extract indice of each labels
        self.label_indices = {label: [i for i, x in enumerate(self.label_list) if x == label] for label in self.labels}

        # few-shot variables
        self.num_support = num_support
        self.num_query = num_query
        self.num_episode = len(self.snr) // ((self.num_support + self.num_query) * len(self.labels))

    def __len__(self):
        return self.num_episode

    def __getitem__(self, idx):
        # idx means index of episode
        sample = dict()

        if self.robust is True:
            for label in self.labels:
                sample[label] = dict()
                label_indices = self.label_indices[label]

                # support set
                support_indices = random.sample(label_indices, self.num_support)
                support_set = [np.concatenate((self.iq[i].transpose(), np.flip(self.iq[i].transpose(), axis=1)), axis=0)
                               for i in support_indices]
                sample[label]['support'] = support_set

                # query set
                query_indices = list(set(label_indices) - set(support_indices))
                query_indices = random.sample(query_indices, self.num_query)
                query_set = [np.concatenate((self.iq[i].transpose(), np.flip(self.iq[i].transpose(), axis=1)), axis=0)
                             for i in query_indices]
                sample[label]['query'] = query_set

        else:
            for label in self.labels:
                sample[label] = dict()
                label_indices = self.label_indices[label]

                # support set
                support_indices = random.sample(label_indices, self.num_support)
                support_set = [self.iq[i].transpose() for i in support_indices]
                sample[label]['support'] = support_set

                # query set
                query_indices = list(set(label_indices) - set(support_indices))
                query_indices = random.sample(query_indices, self.num_query)
                query_set = [self.iq[i].transpose() for i in query_indices]
                sample[label]['query'] = query_set

        return sample