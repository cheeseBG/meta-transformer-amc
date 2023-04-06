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
    def __init__(self, root_path, num_support, num_query, snr_range=None):
        self.root_path = root_path
        self.snr_range = snr_range
        self.config = get_config('config.yaml')

        self.data = h5py.File(os.path.join(self.root_path, "GOLD_XYZ_OSC.0001_1024.hdf5"), 'r')
        self.class_labels = json.load(open(os.path.join(self.root_path, "classes-fixed.json"), 'r'))

        self.iq = self.data['X']
        self.onehot = self.data['Y']
        self.snr = np.squeeze(self.data['Z'])

        self.num_support = num_support
        self.num_query = num_query

        # 클래스 라벨 추출
        self.labels = [int(argwhere(self.onehot[i] == 1)) for i in range(len(self.snr))]

        # Todo
        # 각 라벨별 인덱스 추출
        self.label_indices = {label: [i for i, x in enumerate(self.data) if x[1] == label] for label in self.labels}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 라벨 추출
        label = self.labels[idx]

        # 클래스 인스턴스 추출
        label_instances = [x for x in self.data if x[1] == label]

        # 라벨별 인덱스 리스트 생성
        label_indices = self.label_indices[label]

        # support set 생성
        support_indices = random.sample(label_indices, self.num_support)
        support_set = [label_instances[i] for i in support_indices]

        # query set 생성
        query_indices = list(set(label_indices) - set(support_indices))
        query_indices = random.sample(query_indices, self.num_query)
        query_set = [label_instances[i] for i in query_indices]

        # tensor로 변환
        support_set = torch.stack([x[0] for x in support_set])
        query_set = torch.stack([x[0] for x in query_set])

        # 라벨도 tensor로 변환
        label = torch.tensor(label)

        return support_set, query_set, label