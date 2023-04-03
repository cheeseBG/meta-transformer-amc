import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import config, io_utils

from backbone import TransformerModel
from dataset import TrainSet



def train(params):

    bs = params.batch_sz
    ds = params.dataset_sz
    lr = params.lr
    train_sz = int(ds * params.train_ratio)
    test_sz = ds - train_sz
    n_every_print = ds // bs

    dataset = TrainSet()

    train_set, test_set = torch.utils.data.random_split(dataset, [train_sz, test_sz])
    train_loader = DataLoader(dataset=train_set, batch_size=bs, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=test_sz, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if params.model == 'Transformer':
        net = TransformerModel(n_way=params.n_class, n_feature=64, n_head=8,
                                n_encoder_layers=12, dim_projection=32, dim_feedforward=512).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
                                                                                                 
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("No. of trainable parameters: %d" % pytorch_total_params)

    print("Start training: ")
    for epoch in range(params.stop_epoch):

        running_loss = 0.0
        corrects = 0.0

        net.train()
        for i, data in enumerate(train_loader, 0):
            
            inputs, labels = data

            ## RESHAPE INPUTS
            inputs = np.transpose(inputs, (2, 0, 1))
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimizer
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = F.softmax(outputs, dim=1).argmax(dim=1)
                correct = torch.eq(pred, labels).sum().item() # convert to numpy
                corrects += correct

            # print statistics
            running_loss += loss.item()

            if True:
                print('[Epoch: %d] loss: %.5f, accuracy: %.3f' %
                      (epoch + 1, running_loss / n_every_print, corrects / (n_every_print * params.batch_sz)))
        
        net.eval()
        with torch.no_grad():
            corrects = 0.0
            for i, data in enumerate(test_loader, 0):

                inputs, labels = data

                inputs = np.transpose(inputs, (2, 0, 1))
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                pred = F.softmax(outputs, dim=1).argmax(dim=1)
                corrects = torch.eq(pred, labels).sum().item() # convert to numpy

                print('[Epoch: %d Test] accuracy: %.3f' % (epoch + 1, corrects / test_sz))

    return net
        

if __name__ == '__main__':
    # sets the random seed to a fixed value
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    # obtain parameters
    params = io_utils.parse_args('train')

    # train model 
    model = train(params)
    
    # set checkpoint filename
    checkpoint_dir = io_utils.obtain_checkpoint_dir(config.save_dir, params.model, params.train_aug)
    # save model
    outfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(params.stop_epoch))
    torch.save({'epoch': params.stop_epoch,
                'model_state_dict': model.state_dict(),
                }, outfile)
