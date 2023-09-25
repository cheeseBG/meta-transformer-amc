import os
import h5py
import json
import pickle
import numpy as np
import random
import torch
import torch.utils.data as data
from data.transform import AMCTransform
from runner.utils import get_config
from numpy import argwhere

random.seed(50)


class AMCTrainDataset(data.Dataset):
    def __init__(self, root_path, robust=False, snr_range=None, sample_len=1024):
        super(AMCTrainDataset, self).__init__()

        self.root_path = root_path
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
        self.sample_len = sample_len

        # Sampling data in snr boundary
        if self.snr_range is not None:
            snr_mask = (self.snr_range[0] <= self.snr) & (self.snr <= self.snr_range[1])
            self.iq = self.iq[snr_mask]
            self.onehot = self.onehot[snr_mask]
            self.snr = self.snr[snr_mask]

        # Sampling data which are easy modulation
        mod_mask = np.array([int(argwhere(self.onehot[i] == 1)) in self.config['train_class_indice'] for i in
                             range(len(self.onehot))])
        self.num_modulation = len(self.config['train_class_indice'])

        self.iq = self.iq[mod_mask]
        self.onehot = self.onehot[mod_mask]
        self.snr = self.snr[mod_mask]

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
        x = self.iq[item].transpose()[:, :self.sample_len]

        if self.robust is True:
            revers = np.flip(x.copy(), axis=1)
            x = np.concatenate((x, revers), axis=0)

            sample = {"data": self.transforms(x), "label": label, "snr": self.snr[item]}  # self.transforms(x)
        else:
            x = np.expand_dims(x, axis=1)
            sample = {"data": x, "label": label, "snr": self.snr[item]}

        return sample


class AMCTestDataset(data.Dataset):
    def __init__(self, root_path, robust=False, snr_range=None, sample_len=1024):
        super(AMCTestDataset, self).__init__()

        self.root_path = root_path
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
        self.sample_len = sample_len

        # Sampling data in snr boundary
        if self.snr_range is not None:
            snr_mask = (self.snr_range[0] <= self.snr) & (self.snr <= self.snr_range[1])
            self.iq = self.iq[snr_mask]
            self.onehot = self.onehot[snr_mask]
            self.snr = self.snr[snr_mask]

        mod_mask = np.array([int(argwhere(self.onehot[i] == 1)) in self.config['test_class_indice'] for i in
                             range(len(self.onehot))])
        self.num_modulation = len(self.config['test_class_indice'])

        self.iq = self.iq[mod_mask]
        self.onehot = self.onehot[mod_mask]
        self.snr = self.snr[mod_mask]

        # Sampling test data
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
        x = self.iq[item].transpose()[:, :self.sample_len]

        if self.robust is True:
            revers = np.flip(x.copy(), axis=1)
            x = np.concatenate((x, revers), axis=0)

            sample = {"data": self.transforms(x), "label": label, "snr": self.snr[item]}  # self.transforms(x)
        else:
            x = np.expand_dims(x, axis=1)
            sample = {"data": x, "label": label, "snr": self.snr[item]}

        return sample


class FewShotDataset(data.Dataset):
    def __init__(self, root_path, num_support, num_query, mode='train', robust=False, extension=False, snr_range=None, divide=False, sample_len=1024):
        self.root_path = root_path
        self.robust = robust
        self.extension = extension
        self.snr_range = snr_range
        self.mode = mode
        self.config = get_config('config.yaml')
        self.num_sample = int(4096 * self.config['train_proportion'])  # sample
        self.divide = divide
        self.data = h5py.File(os.path.join(self.root_path, "GOLD_XYZ_OSC.0001_1024.hdf5"), 'r')
        self.class_labels = json.load(open(os.path.join(self.root_path, "classes-fixed.json"), 'r'))
        self.sample_len = sample_len
        self.train_sample_len = self.config['train_sample_size']

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
            mod_mask = np.array([int(argwhere(self.onehot[i] == 1)) in self.config['train_class_indice'] for i in
                                 range(len(self.onehot))])
            self.num_modulation = len(self.config['train_class_indice'])
        elif mode == 'test':
            mod_mask = np.array([int(argwhere(self.onehot[i] == 1)) in self.config['test_class_indice'] for i in
                                 range(len(self.onehot))])
            self.num_modulation = len(self.config['test_class_indice'])
        else:
            print('Mode argument error!')
            exit()

        self.iq = self.iq[mod_mask]
        self.onehot = self.onehot[mod_mask]
        self.snr = self.snr[mod_mask]

        # for extension
        if self.extension is True:
            print("Create amp & phase...")
            self.cmp = self.iq[..., 0] + 1j * self.iq[..., 1]
            self.exts = np.expand_dims(np.abs(self.cmp), axis=1)
            #self.phase = np.expand_dims(np.angle(self.cmp), axis=1)
            #self.exts = np.concatenate([self.amp, self.phase], axis=1)
            print("Done")

        if self.divide is True:
            # Sampling train data
            # each modulation-snr has 4096 I/Q samples
            if mode == 'train':
                sampling_mask = []
                for _ in range(self.num_modulation * len(np.unique(self.snr))):
                    sampling_mask.extend([True for _ in range(self.num_sample)])
                    sampling_mask.extend([False for _ in range(4096 - self.num_sample)])
                sampling_mask = np.array(sampling_mask)
            else:
                sampling_mask = []
                for _ in range(self.num_modulation * len(np.unique(self.snr))):
                    sampling_mask.extend([False for _ in range(4096 - self.num_sample)])
                    sampling_mask.extend([True for _ in range(self.num_sample)])
                sampling_mask = np.array(sampling_mask)

            self.iq = self.iq[sampling_mask]
            self.onehot = self.onehot[sampling_mask]
            self.snr = self.snr[sampling_mask]

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
                # sample[label] = dict()
                # label_indices = self.label_indices[label]

                # # support set
                # support_indices = random.sample(label_indices, self.num_support)
                # # support_set = [np.concatenate((self.iq[i].transpose()[:, :self.sample_len],
                # #                                np.flip(self.iq[i].transpose()[:, :self.sample_len], axis=1)), axis=0)
                # #                for i in support_indices]
                # support_set = [np.concatenate((self.iq[i].transpose()[:, :self.sample_len],
                #                                self.iq[i].transpose()[:, :self.sample_len]), axis=0)
                #                for i in support_indices]
                # sample[label]['support'] = support_set

                # # query set
                # query_indices = list(set(label_indices) - set(support_indices))
                # query_indices = random.sample(query_indices, self.num_query)
                # # query_set = [np.concatenate((self.iq[i].transpose(), np.flip(self.iq[i].transpose(), axis=1)), axis=0)
                # #              for i in query_indices]
                # query_set = [np.concatenate((self.iq[i].transpose(), self.iq[i].transpose()), axis=0) for i in query_indices]
                # sample[label]['query'] = query_set

                sample[label] = dict()
                label_indices = self.label_indices[label]

                # support set
                support_indices = random.sample(label_indices, self.num_support)
                support_set = None
                support_set = [np.concatenate((self.iq[i].transpose()[:, :self.train_sample_len],
                                            self.exts[i][:, :self.train_sample_len]), axis=0)
                            for i in support_indices]
                sample[label]['support'] = support_set

                # query set
                query_indices = list(set(label_indices) - set(support_indices))
                query_indices = random.sample(query_indices, self.num_query)
                query_set = None
                if self.mode == 'train':
                    query_set = [np.concatenate((self.iq[i].transpose()[:, :self.train_sample_len],
                                                self.iq[i].transpose()[:, :self.train_sample_len]), axis=0) for i in query_indices]
                else:
                    query_set = [np.concatenate((self.iq[i].transpose()[:, :self.sample_len],
                                                self.iq[i].transpose()[:, :self.sample_len]), axis=0) for i in query_indices]
                sample[label]['query'] = query_set

        elif self.extension is True:
            for label in self.labels:
                sample[label] = dict()
                label_indices = self.label_indices[label]

                # support set
                support_indices = random.sample(label_indices, self.num_support)
                support_set = None
                support_set = [np.concatenate((self.iq[i].transpose()[:, :self.train_sample_len],
                                            self.exts[i][:, :self.train_sample_len]), axis=0)
                            for i in support_indices]
                sample[label]['support'] = support_set

                # query set
                query_indices = list(set(label_indices) - set(support_indices))
                query_indices = random.sample(query_indices, self.num_query)
                query_set = None
                if self.mode == 'train':
                    query_set = [np.concatenate((self.iq[i].transpose()[:, :self.train_sample_len],
                                                self.exts[i][:, :self.train_sample_len]), axis=0) for i in query_indices]
                else:
                    # query_set = [np.concatenate((self.iq[i].transpose()[:, :self.sample_len],
                    #                            self.exts[i][:, :self.sample_len]), axis=0) for i in query_indices]
                     # self-duplicate
                    num_dup = (self.train_sample_len // self.sample_len)
                    query_set = [np.concatenate([np.concatenate((self.iq[i].transpose()[:, :self.sample_len],
                                               self.exts[i][:, :self.sample_len]), axis=0) for _ in range(num_dup)], axis=1) for i in query_indices]
                sample[label]['query'] = query_set

        else:
            for label in self.labels:
                sample[label] = dict()
                label_indices = self.label_indices[label]

                # support set
                support_indices = random.sample(label_indices, self.num_support)
                support_set = [self.iq[i].transpose()[:, :self.train_sample_len] for i in support_indices]
                sample[label]['support'] = support_set

                # query set
                query_indices = list(set(label_indices) - set(support_indices))
                query_indices = random.sample(query_indices, self.num_query)
                query_set = None
                if self.mode == 'train':
                    query_set = [self.iq[i].transpose()[:, :self.train_sample_len] for i in query_indices]
                else:
                    # zero padding
                    query_set = [np.concatenate((self.iq[i].transpose()[:, :self.sample_len], 
                                                 np.zeros((2, self.train_sample_len-self.sample_len),dtype=np.float32)), axis=1) for i in query_indices]
                                                 
                    # self-duplicate
                    num_dup = (self.train_sample_len // self.sample_len)
                    query_set = [np.concatenate([self.iq[i].transpose()[:, :self.sample_len] for _ in range(num_dup)], axis=1) for i in query_indices]
                sample[label]['query'] = query_set
        return sample


class FewShotDatasetForOnce(data.Dataset):
    def __init__(self, root_path, num_support, num_query, robust=False, snr_range=None, sample_len=1024):
        self.root_path = root_path
        self.robust = robust
        self.snr_range = snr_range
        self.config = get_config('config.yaml')
        self.sample_len = sample_len

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
        mod_mask = np.array([int(argwhere(self.onehot[i] == 1)) in self.config['test_class_indice'] for i in range(len(self.onehot))])
        self.num_modulation = len(self.config['test_class_indice'])

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
                support_set = [np.concatenate((self.iq[i].transpose()[:, :self.sample_len],
                                               np.flip(self.iq[i].transpose()[:, :self.sample_len], axis=1)), axis=0)
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


'''
RML2016.10A
'''
class FewShotDataset2016(data.Dataset):
    def __init__(self, root_path, num_support, num_query, mode='train', robust=False, extension=False, snr_range=None, divide=False, sample_len=128):
        self.root_path = root_path
        self.robust = robust
        self.extension = extension
        self.snr_range = snr_range
        self.mode = mode
        self.config = get_config('config.yaml')
        self.num_sample = int(1000 * self.config['train_proportion'])  # sample
        self.divide = divide
        self.labels = list()
        if mode == 'train':
            self.labels = [self.config['total_class16'][idx] for idx in self.config['train_class_indice16']]
        elif mode == 'test':
           self.labels = [self.config['total_class16'][idx] for idx in self.config['test_class_indice16']]
        self.num_modulation = len(self.labels)

        with open(self.root_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data_dict = u.load()

        self.data = data_dict
        self.sample_len = sample_len
        #self.train_sample_len = self.config['train_sample_size']
        self.train_sample_len = 128

        self.iq = None
        self.onehot = None
        # Sampling data in snr boundary
        for mod_idx, mod in enumerate(self.labels):
            for snr in range(self.snr_range[0], self.snr_range[1], 2):
                onehot = np.zeros(self.num_modulation)
                onehot[mod_idx] = 1

                if self.iq != None:
                    if self.divide is True:
                        if mode == 'train':
                            self.iq.vstack((self.iq, self.data[(mod, snr)[:self.num_sample]]))
                            onehot_arr = np.array([onehot for _ in range(self.num_sample)])
                            self.onehot.vstack((self.onehot, onehot_arr))
                        elif mode == 'test':
                            self.iq.vstack((self.iq, self.data[(mod, snr)[self.num_sample:]]))
                            onehot_arr = np.array([onehot for _ in range(1000-self.num_sample)])
                            self.onehot.vstack((self.onehot, onehot_arr))
                    else:
                        self.iq.vstack((self.iq, self.data[(mod, snr)]))
                        onehot_arr = np.array([onehot for _ in range(1000)])
                        self.onehot.vstack((self.onehot, onehot_arr))
                else:
                    if self.divide is True:
                        if mode == 'train':
                            self.iq = self.data[(mod, snr)[:self.num_sample]]
                            self.onehot = np.array([onehot for _ in range(self.num_sample)])
                        elif mode == 'test':
                            self.iq = self.data[(mod, snr)[self.num_sample:]]
                            self.onehot = np.array([onehot for _ in range(1000-self.num_sample)])
                    else:
                        self.iq = self.data[(mod, snr)]
                        self.onehot = np.array([onehot for _ in range(1000)])
  
        # for extension
        if self.extension is True:
            print("Create amp & phase...")
            self.cmp = self.iq[..., 0] + 1j * self.iq[..., 1]
            self.exts = np.expand_dims(np.abs(self.cmp), axis=1)
            #self.phase = np.expand_dims(np.angle(self.cmp), axis=1)
            #self.exts = np.concatenate([self.amp, self.phase], axis=1)
            print("Done")

        # Extract indice of each labels
        self.label_indices = {label: [i for i, x in enumerate(self.label_list) if x == label] for label in self.labels}

        # few-shot variables
        self.num_support = num_support
        self.num_query = num_query
        self.num_episode = len(self.iq) // ((self.num_support + self.num_query) * len(self.labels))

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
                # support_set = [np.concatenate((self.iq[i].transpose()[:, :self.sample_len],
                #                                np.flip(self.iq[i].transpose()[:, :self.sample_len], axis=1)), axis=0)
                #                for i in support_indices]
                support_set = [np.concatenate((self.iq[i][:, :self.sample_len],
                                               self.iq[i][:, :self.sample_len]), axis=0)
                               for i in support_indices]
                sample[label]['support'] = support_set

                # query set
                query_indices = list(set(label_indices) - set(support_indices))
                query_indices = random.sample(query_indices, self.num_query)
                # query_set = [np.concatenate((self.iq[i].transpose(), np.flip(self.iq[i].transpose(), axis=1)), axis=0)
                #              for i in query_indices]
                query_set = [np.concatenate((self.iq[i], self.iq[i]), axis=0) for i in query_indices]
                sample[label]['query'] = query_set

        elif self.extension is True:
            for label in self.labels:
                sample[label] = dict()
                label_indices = self.label_indices[label]

                # support set
                support_indices = random.sample(label_indices, self.num_support)
                support_set = None
                support_set = [np.concatenate((self.iq[i][:, :self.train_sample_len],
                                            self.exts[i][:, :self.train_sample_len]), axis=0)
                            for i in support_indices]
                sample[label]['support'] = support_set

                # query set
                query_indices = list(set(label_indices) - set(support_indices))
                query_indices = random.sample(query_indices, self.num_query)
                query_set = None
                if self.mode == 'train':
                    query_set = [np.concatenate((self.iq[i][:, :self.train_sample_len],
                                                self.exts[i][:, :self.train_sample_len]), axis=0) for i in query_indices]
                else:
                    # query_set = [np.concatenate((self.iq[i].transpose()[:, :self.sample_len],
                    #                            self.exts[i][:, :self.sample_len]), axis=0) for i in query_indices]
                     # self-duplicate
                    num_dup = (self.train_sample_len // self.sample_len)
                    query_set = [np.concatenate([np.concatenate((self.iq[i][:, :self.sample_len],
                                               self.exts[i][:, :self.sample_len]), axis=0) for _ in range(num_dup)], axis=1) for i in query_indices]
                sample[label]['query'] = query_set

        else:
            for label in self.labels:
                sample[label] = dict()
                label_indices = self.label_indices[label]

                # support set
                support_indices = random.sample(label_indices, self.num_support)
                support_set = [self.iq[i][:, :self.train_sample_len] for i in support_indices]
                sample[label]['support'] = support_set

                # query set
                query_indices = list(set(label_indices) - set(support_indices))
                query_indices = random.sample(query_indices, self.num_query)
                query_set = None
                if self.mode == 'train':
                    query_set = [self.iq[i][:, :self.train_sample_len] for i in query_indices]
                else:
                    # zero padding
                    query_set = [np.concatenate((self.iq[i][:, :self.sample_len], 
                                                 np.zeros((2, self.train_sample_len-self.sample_len),dtype=np.float32)), axis=1) for i in query_indices]
                                                 
                    # self-duplicate
                    num_dup = (self.train_sample_len // self.sample_len)
                    query_set = [np.concatenate([self.iq[i][:, :self.sample_len] for _ in range(num_dup)], axis=1) for i in query_indices]
                sample[label]['query'] = query_set
        return sample