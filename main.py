'''
Todo:
    1. Read_csi_csv function
    2. Automatically check number of query to use
'''


import numpy as np
from scipy.io import loadmat
import multiprocessing as mp
import os
import torch
import torch.optim as optim

from util import read_csi, read_csi_from_pcap, read_csi_csv
from proto import load_protonet_conv
from train import train
from test import test
from config import param

# from sklearn.model_selection import train_test_split

train_mod = False
use_pretrain = True
test_mod = True


##### Load the dataset
data_folder = 'm1c4_PCA_test_80'
train_env = 'A1'
# train_folder_name = 'few_shot_datasets/ReWis/' + data_folder + '/train_A1'  # ReWis dataset
# train_folder_name = 'few_shot_datasets/pcap/'  # New dataset
train_folder_name = 'few_shot_datasets/jji_home/'  # New dataset

test_env = 'A3'
# test_folder_name = 'few_shot_datasets/ReWis/' + data_folder + '/test_' + test_env  # ReWis dataset
# test_folder_name = 'few_shot_datasets/pcap/'  # New dataset
test_folder_name = 'few_shot_datasets/jji_home/'  # New dataset

##### Train Phase
model_out_name = 'pretrain/model_s{}q{}.pt'.format(str(param['train_support']), str(param['train_query']))

if train_mod is True:
    train_x, train_y = read_csi_csv(train_folder_name, one_file=True)
    # train_x, train_y = read_csi_from_pcap(train_folder_name)
    train_x = np.expand_dims(train_x, axis=1)

    # sample_example = extract_sample(2, 8, 5, train_x, train_y)
    model = load_protonet_conv(
        x_dim=(1, 512, 256),
        hid_dim=64,
        z_dim=64,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    max_epoch = 2
    epoch_size = 200

    train(model, optimizer, train_x, train_y,
          param['train_way'], param['train_support'], param['train_query'],
          max_epoch, epoch_size)

    # Save Model
    torch.save(model.state_dict(), model_out_name)
#####################

##### Test Phase

if test_mod is True:
    if use_pretrain is True:
        model = load_protonet_conv(
            x_dim=(1, 512, 256),
            hid_dim=64,
            z_dim=64,
        )
        model.load_state_dict(torch.load(model_out_name))

    print('Read Dataset....')
    test_x, test_y = read_csi_from_pcap(test_folder_name)
    # test_x, test_y = read_csi_csv(train_folder_name, one_file=True)
    print('Done')
    test_x = np.expand_dims(test_x, axis=1)

    # ReWis test
    # test_env = 'A2'
    # test_folder_name = 'few_shot_datasets/ReWis/' + data_folder + '/test_' + test_env
    # test_x, test_y = read_csi(test_folder_name)
    # test_x = np.expand_dims(test_x, axis=1)

    result_out_name = 'results/result_A1' + test_env + '_' + data_folder + '.pt'

    test_episode = 1
    print(data_folder + ': trained on ' + train_env + ', testing on ' + test_env)
    CF, acc = test(model, test_x, test_y,
                   param['test_way'], param['test_support'], param['test_query'],
                   test_episode)