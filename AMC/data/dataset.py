import torch
from data.transform import AMCTransform
import torch.utils.data as data
import os
import h5py
import json
from numpy import argwhere
import numpy as np


class AMCDataset(data.Dataset):
    def __init__(self, root_path, pre=None, mode=None, snr_range=None):
        super(AMCDataset, self).__init__()

        self.root_path = root_path
        self.pre = pre
        self.mode = mode
        self.snr_range = snr_range
        self.transforms = AMCTransform()

        self.data = h5py.File(os.path.join(self.root_path, "GOLD_XYZ_OSC.0001_1024.hdf5"), 'r')
        self.class_labels = json.load(open(os.path.join(self.root_path, "classes-fixed.json"), 'r'))

        self.iq = self.data['X']
        self.onehot = self.data['Y']
        self.snr = np.squeeze(self.data['Z'])

        if self.snr_range is not None:
            snr_mask = (self.snr_range[0] <= self.snr) & (self.snr <= self.snr_range[1])
            self.iq = self.iq[snr_mask]
            self.onehot = self.onehot[snr_mask]
            self.snr = self.snr[snr_mask]

    def __len__(self):
        return self.iq.shape[0]

    def __getitem__(self, item):
        label = int(argwhere(self.onehot[item] == 1))
        # print(label)
        # print(self.snr[item])
        x = self.iq[item].transpose()
        revers = np.flip(x.copy(), axis=1)
        x = np.concatenate((x, revers), axis=0)

        sample = {"data": self.transforms(x), "label": label, "snr": self.snr[item]}  # self.transforms(x)

        return sample
