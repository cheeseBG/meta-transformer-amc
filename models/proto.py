import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from models.robustcnn import *
from models.vit import *
from models.protonet import *
from models.daelstm import *
from models.resnet import *


class ProtoNet(nn.Module):
    def __init__(self, encoder, config):
        """
        Args:
            encoder : Feture extractor(model) encoding the dataloader dataframes in sample
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
        """
        super(ProtoNet, self).__init__()
        self.encoder = encoder.cuda(0)
        self.config = config

    def proto_train(self, sample):
        n_way = len(sample.keys())
        n_support = self.config['num_support']
        n_query = self.config['num_query']

        """
        support shape: [K_way, num_support, 1, I/Q, data_length]
        query shape: [K_way, num_query, 1, I/Q, data_length]
        """
        x_support = None
        x_query = None
        for label in sample.keys():
            if x_support is None:
                x_support = np.array([np.array(iq) for iq in sample[label]['support']])
            else:
                x_support = np.vstack([x_support, np.array([np.array(iq) for iq in sample[label]['support']])])
            if x_query is None:
                x_query = np.array([np.array(iq) for iq in sample[label]['query']])
            else:
                x_query = np.vstack([x_query, np.array([np.array(iq) for iq in sample[label]['query']])])

        x_support = torch.from_numpy(x_support).cuda(0)
        x_query = torch.from_numpy(x_query).cuda(0)

        if self.config['model'] == 'resnet':
            x_support = x_support.reshape((-1,2,1,x_support.shape[-1]))
            x_query = x_query.reshape((-1,2,1,x_query.shape[-1]))

        # target indices are 0 ... n_way-1
        target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.cuda(0)
        
        if self.config['model'] in ['lstm', 'daelstm']:
            x_support = x_support.squeeze().permute(0, 2, 1)
            x_query = x_query.squeeze().permute(0, 2, 1)

        # encode dataloader dataframes of the support and the query set
        z_support = self.encoder.forward(x_support)
        z_query = self.encoder.forward(x_query)
        z_support_dim = z_support.size(-1)
        z_proto = z_support.view(n_way, n_support, z_support_dim).mean(1)

        # compute distances
        dists = torch.cdist(z_query, z_proto)

        # compute probabilities
        log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat
        }

    def create_protoNet(self, sample):
        n_way = len(sample.keys())
        n_support = self.config['num_support']

        """
        support shape: [K_way, num_support, 1, I/Q, data_length]
        query shape: [K_way, num_query, 1, I/Q, data_length]
        """
        x_support = None
        x_query = None
        for label in sample.keys():
            if x_support is None:
                x_support = np.array([np.array(iq) for iq in sample[label]['support']])
            else:
                x_support = np.vstack([x_support, np.array([np.array(iq) for iq in sample[label]['support']])])
            if x_query is None:
                x_query = np.array([np.array(iq) for iq in sample[label]['query']])
            else:
                x_query = np.vstack([x_query, np.array([np.array(iq) for iq in sample[label]['query']])])

        x_support = torch.from_numpy(x_support).cuda(0)

        # encode dataloader dataframes of the support and the query set
        z_support = self.encoder.forward(x_support)
        z_support_dim = z_support.size(-1)
        z_proto = z_support.view(n_way, n_support, z_support_dim).mean(1)

        return z_proto


    def proto_test(self, sample):
        n_way = len(sample.keys())
        n_support = self.config['num_support']
        n_query = self.config['num_query']

        """
        support shape: [K_way, num_support, 1, I/Q, data_length]
        query shape: [K_way, num_query, 1, I/Q, data_length]
        """
        x_support = None
        x_query = None
        for label in sample.keys():
            if x_support is None:
                x_support = np.array([np.array(iq) for iq in sample[label]['support']])
            else:
                x_support = np.vstack([x_support, np.array([np.array(iq) for iq in sample[label]['support']])])
            if x_query is None:
                x_query = np.array([np.array(iq) for iq in sample[label]['query']])
            else:
                x_query = np.vstack([x_query, np.array([np.array(iq) for iq in sample[label]['query']])])

        x_support = torch.from_numpy(x_support).cuda(0)
        x_query = torch.from_numpy(x_query).cuda(0)

        # target indices are 0 ... n_way-1
        target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.cuda(0)

        if self.config['model'] == 'daelstm':
            x_support = x_support.squeeze().permute(0, 2, 1)
            x_query = x_query.squeeze().permute(0, 2, 1)

        # encode dataloader dataframes of the support and the query set
        z_support = self.encoder.forward(x_support)
        z_query = self.encoder.forward(x_query)
        z_support_dim = z_support.size(-1)
        z_proto = z_support.view(n_way, n_support, z_support_dim).mean(1)

        dists = torch.cdist(z_query, z_proto)

        # compute probabilities
        log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
        _, y_hat = log_p_y.max(2)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()  # y_hat과 gt 같은지 비교

        return {
            'acc': acc_val.item(),
            'y_hat': y_hat
        }


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def load_protonet_conv(**kwargs):
    """
    Loads the prototypical network model
    Arg:
        x_dim (tuple): dimension of input dataloader dataframes
        hid_dim (int): dimension of hidden layers in conv blocks
        z_dim (int): dimension of embedded dataloader dataframes
    Returns:
        Model (Class ProtoNet)
    """
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    config = kwargs['config']

    encoder = ProtoNet_CNN(x_dim[0], hid_dim, z_dim)
    
    return ProtoNet(encoder, config)


def load_protonet_robustcnn(config):
    encoder = nn.Sequential(
        ABlock(),
        BBlock(),
        nn.AvgPool2d(2, 1),
        CBlock1(),
        CBlock2(),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten()
    )

    return ProtoNet(encoder, config)


def load_protonet_vit(config, model_params):

    encoder = ViT(
        in_channels=model_params["in_channels"],
        patch_size=model_params["patch_size"],
        embed_dim=model_params["embed_dim"],
        num_layers=model_params["num_layers"],
        num_heads=model_params["num_heads"],
        mlp_dim=model_params["mlp_dim"],
        num_classes=model_params["num_classes"],
        in_size=model_params["in_size"]

    )
    return ProtoNet(encoder, config)

def load_protonet_daelstm(config):

    encoder = DAELSTM(input_shape=[1,2,1024],
                   modulation_num=config["num_classes"])

    return ProtoNet(encoder, config)

