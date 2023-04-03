#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the embeding using training data from environment A_1
Resnet12 is used as the base model
the trained network is saved and is used for extracting embeddings for training protonet
"""
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from scipy.io import loadmat
import multiprocessing as mp
import os
from resnet import resnet12
from sklearn.model_selection import train_test_split


def read_mat(csi_directory_path, csi_action):
    """
    Reads all the actions from a given alphabet_directory
    """
    datax = []
    datay = []

    csi_mats = os.listdir(csi_directory_path)
    for csi_mat in csi_mats:
        mat = loadmat(csi_directory_path + csi_mat)
        if 'PCA' in csi_directory_path:
            data = mat['cfm_data']
        else:
            data = mat['iq_data']

        datax.extend([data])
        datay.extend([csi_action])
    return np.array(datax), np.array(datay)

def read_csi(base_directory):
    """
    Reads all the alphabets from the base_directory
    Uses multithreading to decrease the reading time drasticallytrain_x
    """
    datax = None
    datay = None
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply(read_mat,args=(
                              base_directory + '/' + directory + '/', directory, 
                              )) for directory in os.listdir(base_directory)]
    pool.close()
    for result in results:
        if datax is None:
            datax = result[0]
            datay = result[1]
        else:
            datax = np.vstack([datax, result[0]])
            datay = np.concatenate([datay, result[1]])
    return datax, datay

data_folder = 'extractd_3x4/m1c4_PCA_80_300_extracted_3x4'
train_folder_name = 'few_shot_datasets/' + data_folder + '/train_A1'

model_out_name = 'models/model_' + data_folder.split('/')[-1]  + '.pt'


trainx, trainy = read_csi(train_folder_name)
trainx = np.expand_dims(trainx, axis=1)


for i in range(trainy.shape[0]):
    if trainy[i]=='walk':
        trainy[i] = 0
    elif trainy[i]=='empty':
        trainy[i] = 1
    elif trainy[i] == 'jump':
        trainy[i] = 2
    else:
        trainy[i] = 3
# no train test
# tain_x = trainx
# train_y = trainy
 
train_x, test_x, train_y, test_y = train_test_split(trainx, trainy, test_size = 0.1, random_state=40, shuffle=True)


datax  = torch.from_numpy(train_x).float()

# converting the target into torch format
train_y = train_y.astype(int);
datay = torch.from_numpy(train_y)


# converting validation images into torch format
test_x  = torch.from_numpy(test_x).float()

# converting the target into torch format
test_y = test_y.astype(int);
test_y = torch.from_numpy(test_y)

def extract_batch(datax, datay, batch_size = 5):
    num_batch = torch.floor(torch.tensor(datax.shape[0]/batch_size)).int()
    batch_x = torch.zeros(num_batch, batch_size, datax.shape[1],datax.shape[2],datax.shape[3])
    batch_y = torch.zeros(num_batch, batch_size)
    for i in range(num_batch):
        x = datax[i*batch_size:(i+1)*batch_size,:,:,:]
        y = datay[i*batch_size:(i+1)*batch_size]
        batch_x[i,:,:,:,:] = x
        batch_y[i,:] = y
    return (num_batch, batch_x, batch_y)
num_batch, batch_x, batch_y = extract_batch(datax, datay, batch_size = 5)
# %%
epochs = 100
lr_decay_epochs = '60,80'
lr_decay_rate = 0.1
learning_rate = 0.05
n_gpu = 1

iterations = lr_decay_epochs.split(',')
lr_decay_epochs = list([])
for it in iterations:
    lr_decay_epochs.append(int(it))
# %%
model = resnet12(avg_pool=True, num_classes = 4)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.0005)
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    if n_gpu > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    criterion = criterion.cuda()
    cudnn.benchmark = True
   
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        # softmax = torch.exp(output).cpu()
        # prob = list(softmax.numpy())
        # pr = np.argmax(prob, axis=1)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(epoch, datax, datay, model, criterion, optimizer):
    """One epoch training"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    num_batch, batch_x, batch_y = extract_batch(datax, datay, batch_size = 5)

    for idx in range(num_batch):

        input = batch_x[idx,:,:,:,:].squeeze().unsqueeze(1)
        target = batch_y[idx,:].squeeze().long()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), input.size(0))
        # top1.update(acc1[0], input.size(0))
        # top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters====================

    

    # return top1.avg, losses.avg
    # %%
for epoch in range(1, epochs + 1):    
    
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    train(epoch, datax, datay, model, criterion, optimizer)
 
torch.save(model.state_dict(), model_out_name)
