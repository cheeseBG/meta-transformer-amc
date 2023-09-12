import os
import h5py
import json
import random
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from numpy import argwhere
from data.transform import AMCTransform
from runner.utils import get_config

random.seed(50)

'''
A: Low SNR
B: High SNR
'''
class AMCDataset(data.Dataset):
    def __init__(self, data_path, mode='train', snr_A=-20, snr_B=20, label='OOK'):
        super(AMCDataset, self).__init__()

        self.data_path = data_path
        self.snr_A = snr_A
        self.snr_B = snr_B
        self.mode = mode
        self.label = label
        self.transforms = AMCTransform()
        self.config = get_config('config.yaml')

        self.data = h5py.File(os.path.join(self.data_path, "GOLD_XYZ_OSC.0001_1024.hdf5"), 'r')
        self.class_labels = json.load(open(os.path.join(self.data_path, "classes-fixed.json"), 'r'))

        self.iq = self.data['X']
        self.onehot = self.data['Y']
        self.snr = np.squeeze(self.data['Z'])
        self.num_modulation = 24
        self.num_sample = int(4096 * self.config['train_proportion'])  # sample per modulation-snr

        if self.mode == 'test':
            self.num_sample = 4096 - int(4096 * self.config['train_proportion'])  # sample per modulation-snr
        
        # Sampling modulation label
        label_idx = self.config['total_class'].index(self.label)
        label_mask = [False for _ in range(len(self.onehot))]
        for i in range(4096*26):
            label_mask[label_idx*(4096*26) +  i] = True
        self.iq[label_mask]
        self.onehot[label_mask]
        self.snr[label_mask]

        # Sampling SNR A
        snr_A_mask = (self.snr_A == self.snr)
        self.iq_A = self.iq[snr_A_mask]
        self.onehot_A = self.onehot[snr_A_mask]
        self.snr_A = self.snr[snr_A_mask]

        # Sampling SNR B
        snr_B_mask = (self.snr_B == self.snr)
        self.iq_B = self.iq[snr_B_mask]
        self.onehot_B = self.onehot[snr_B_mask]
        self.snr_B = self.snr[snr_B_mask]

        # Sampling train data
        # each modulation-snr has 4096 I/Q samples
        sampling_mask = []
        for _ in range(self.num_modulation * len(np.unique(self.snr_A))):
            if self.mode == 'train':
                sampling_mask.extend([True for _ in range(self.num_sample)])
                sampling_mask.extend([False for _ in range(4096-self.num_sample)])
            else:
                sampling_mask.extend([False for _ in range(4096 - self.num_sample)])
                sampling_mask.extend([True for _ in range(self.num_sample)])
        sampling_mask = np.array(sampling_mask)

        self.iq_A = self.iq_A[sampling_mask]
        self.onehot_A = self.onehot_A[sampling_mask]
        self.snr_A = self.snr_A[sampling_mask]

        self.iq_B = self.iq_B[sampling_mask]
        self.onehot_B = self.onehot_B[sampling_mask]
        self.snr_B = self.snr_B[sampling_mask]

    def __len__(self):
        return self.iq_A.shape[0]

    def __getitem__(self, item):
        label_A = int(argwhere(self.onehot_A[item] == 1))
        x_a = self.iq_A[item].transpose()
        x_a = np.expand_dims(x_a, axis=0)
        sample_A = {"data": x_a, "label": label_A, "snr": self.snr_A[item]}

        label_B = int(argwhere(self.onehot_B[item] == 1))
        x_b = self.iq_B[item].transpose()
        x_b = np.expand_dims(x_b, axis=0)
        sample_B = {"data": x_b, "label": label_B, "snr": self.snr_B[item]}

        return sample_A, sample_B


if __name__ == '__main__':
    train_data = AMCDataset('../amc_dataset/RML2018')
    test_data = AMCDataset('../amc_dataset/RML2018', mode='test')
    
    print(train_data.__getitem__(0)[0]['data'].shape)
    print(train_data.__len__())