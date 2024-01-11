import os
import yaml
import torch
import logging
import numpy as np
import pandas as pd
import random
import importlib
import inspect
import matplotlib.pyplot as plt

# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def get_function_arguments(func):
    args = inspect.getfullargspec(func).args
    return args

def model_selection(config, model_params, mode='train'):
    model_name = config['model']

    supervised_models = {
        'robustcnn': {'module': 'models.robustcnn', 'class': 'RobustCNN', 'optimizer': torch.optim.SGD},
        'resnet': {'module': 'models.resnet', 'class': 'ResNetStack', 'optimizer': torch.optim.Adam},
        'daelstm_super': {'module': 'models.daelstm', 'class': 'DAELSTM', 
                          'input_shape': [1, 2, 1024], 'modulation_num': 24, 'optimizer': torch.optim.Adam},
    }

    meta_models = {
        'vit_main': {'module': 'models.proto', 'class': 'load_protonet_vit', 
                     'config': config, 'model_params': model_params, 'optimizer': torch.optim.Adam},
        'vit_sub': {'module': 'models.proto', 'class': 'load_protonet_vit', 
                    'config': config, 'model_params': model_params, 'optimizer': torch.optim.Adam},
        'protonet': {'module': 'models.proto', 'class': 'load_protonet_conv', 
                     'x_dim': (1, 512, 256), 'hid_dim': 32, 'z_dim': 24, 'config': config, 'optimizer': torch.optim.Adam},
        'daelstm_meta': {'module': 'models.proto', 'class': 'load_protonet_daelstm', 
                         'config': config, 'optimizer': torch.optim.Adam},
    }

    if model_name in supervised_models:
        model_info = supervised_models[model_name]
    elif model_name in meta_models:
        model_info = meta_models[model_name]
    else:
        raise NotImplementedError(model_name)

    module = importlib.import_module(model_info['module'])
    model_class = getattr(module, model_info['class'])

    if 'input_shape' in model_info:
        net = model_class(input_shape=model_info['input_shape'], modulation_num=model_info['modulation_num'])
    elif model_params['lr_mode'] == 'supervised':
        relevant_args = {key: model_info[key] for key in model_class.__init__.__code__.co_varnames if key in model_info}
        net = model_class(**relevant_args)
    else:
        function_args = get_function_arguments(model_class)
        relevant_args = {key: model_info[key] for key in function_args if key in model_info}
        net = model_class(**relevant_args)

    if mode == 'train':
        optimizer = model_info['optimizer'](net.parameters(), lr=model_params['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=model_params['lr_gamma'])

        return net, optimizer, scheduler
    else:
        return net

def euclidean_dist(x, y):
    """
    Computes euclidean distance btw x and y
    Args:
        x (torch.Tensor): shape (n, d). n usually n_way*n_query
        y (torch.Tensor): shape (m, d). m usually n_way
    Returns:
        torch.Tensor: shape(n, m). For each query, the distances to each centroid
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: yellow + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def result2csv(result_list, size_list, save_path):
    tmp_dict = dict()
    for i, size in enumerate(size_list):
        tmp_dict[size] = result_list[i]
    df = pd.DataFrame(tmp_dict)
    df.to_csv(os.path.join(save_path, 'result.csv'), index=False)
