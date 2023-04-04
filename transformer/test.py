import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import io_utils, config
from backbone import TransformerModel
from amc_dataset import TestSet

def test(params):
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if params.model == 'Transformer':
        model = TransformerModel(n_way=params.n_way, n_feature=64, n_head=8,
                                n_encoder_layers=6, dim_projection=32, dim_feedforward=512).to(device)
    
    checkpoint_dir = io_utils.obtain_checkpoint_dir(config.save_dir, params.model)
    path = os.path.join(checkpoint_dir, '{:d}.tar'.format(params.stop_epoch))
    state_dict = torch.load(path)['model_state_dict']
    state_dict.pop('classifier.L.weight_g', None)
    state_dict.pop('classifier.L.weight_v', None)
    model.load_state_dict(state_dict, strict=False)

    # initialize amc_dataset
    test_data = TestSet(params.n_way, params.k_shot, params.k_query, params.win_size)
    
    x_spt, y_spt, x_qry, y_qry = test_data.load_test_set()
    x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), \
                                 x_qry.to(device), y_qry.to(device)

    for name, param in model.named_parameters():
        if name != 'classifier.L.weight_g' and name != 'classifier.L.weight_v':
            param.requires_grad = False
    
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    qrysz = params.k_query * params.n_way

    model.train()
    for epoch in range(2000):
        
        logits = model(x_spt)
        loss = criterion(logits, y_spt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 - 1 == 0:
            confusion_matrix = torch.zeros(params.n_way, params.n_way)
            with torch.no_grad():

                # [setsz, nway]
                logits_q = model(x_qry)
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)

                # scalar
                correct = torch.eq(pred_q, y_qry).sum().item() / qrysz

                for t, p in zip(y_qry.view(-1), pred_q.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                accuracy_each_class = confusion_matrix.diag()/confusion_matrix.sum(1)
                print(accuracy_each_class)
                std = torch.std(accuracy_each_class)
        
    return correct, std

if __name__ == '__main__':

    # sets the random seed to a fixed value
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(220)

    # obtain parameters
    params = io_utils.parse_args('test')

    acc, std = test(params)
    print(acc, std)
