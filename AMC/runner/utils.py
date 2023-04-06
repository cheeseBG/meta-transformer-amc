import yaml
from models.robustcnn import RobustCNN


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


# get model params in config
def model_selection(model_name):
    if model_name == 'robustcnn':
        return RobustCNN(n_class=24, softmax=False)
    else:
        raise NotImplementedError(model_name)

# import os
# import numpy as np
# import multiprocessing as mp
# import torch
# import pandas as pd
# from config import param
#
# def extract_train_sample(n_way, n_support, n_query, datax, datay):
#     """
#     Picks random sample of size n_support+n_querry, for n_way classes
#     Args:
#         n_way (int): number of classes in a classification task
#         n_support (int): number of labeled examples per class in the support set
#         n_query (int): number of labeled examples per class in the query set
#         datax (np.array): amc_dataset of dataloader dataframes
#         datay (np.array): amc_dataset of labels
#     Returns:
#         (dict) of:
#           (torch.Tensor): sample of dataloader dataframes. Size (n_way, n_support+n_query, (dim))
#           (int): n_way
#           (int): n_support
#           (int): n_query
#     """
#     sample = None
#     K = np.random.choice(np.unique(datay), n_way, replace=False)
#
#     for cls in K:
#         datax_cls = datax[datay == cls]
#         perm = np.random.permutation(datax_cls)
#         sample_cls = perm[:(n_support + n_query)]
#         if sample is None:
#             sample = np.array([sample_cls])
#         else:
#             sample = np.vstack([sample, [np.array(sample_cls)]])
#         #sample.append(sample_cls)
#
#     sample = np.array(sample)
#     sample = torch.from_numpy(sample).float()
#
#     # sample = sample.permute(0,1,4,2,3)
#     # sample = np.expand_dims(sample, axis= 0)
#     return ({
#         'csi_mats': sample,
#         'n_way': n_way,
#         'n_support': n_support,
#         'n_query': n_query
#     })
#
#
# def extract_test_sample(n_way, n_support, n_query, datax, datay):
#     """
#     Picks random sample of size n_support+n_querry, for n_way classes
#     Args:
#         n_way (int): number of classes in a classification task
#         n_support (int): number of labeled examples per class in the support set
#         n_query (int): number of labeled examples per class in the query set
#         datax (np.array): amc_dataset of dataloader dataframes
#         datay (np.array): amc_dataset of labels
#     Returns:
#         (dict) of:
#           (torch.Tensor): sample of dataloader dataframes. Size (n_way, n_support+n_query, (dim))
#           (int): n_way
#           (int): n_support
#           (int): n_query
#     """
#     K = np.array(param['test_labels'])
#
#     # extract support set & query set
#     support_sample = []
#     query_sample = []
#     for cls in K:
#         datax_cls = datax[datay == cls]
#         support_cls = datax_cls[:n_support]
#         query_cls = np.array(datax_cls[n_support:n_support+n_query])
#         support_sample.append(support_cls)
#         query_sample.append(query_cls)
#
#     support_sample = np.array(support_sample)
#     query_sample = np.array(query_sample)
#     support_sample = torch.from_numpy(support_sample).float()
#     query_sample = torch.from_numpy(query_sample).float()
#
#     return ({
#         'support': support_sample,
#         'query': query_sample,
#         'n_way': n_way,
#         'n_support': n_support,
#         'n_query': n_query
#     })
#
#
# def euclidean_dist(x, y):
#     """
#     Computes euclidean distance btw x and y
#     Args:
#         x (torch.Tensor): shape (n, d). n usually n_way*n_query
#         y (torch.Tensor): shape (m, d). m usually n_way
#     Returns:
#         torch.Tensor: shape(n, m). For each query, the distances to each centroid
#     """
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     assert d == y.size(1)
#
#     x = x.unsqueeze(1).expand(n, m, d)
#     y = y.unsqueeze(0).expand(n, m, d)
#
#     return torch.pow(x - y, 2).sum(2)
#

# def cosine_similarity(x, y):
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     assert d == y.size(1)



