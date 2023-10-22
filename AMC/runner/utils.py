import yaml
import torch
import logging
import wandb
from models.robustcnn import RobustCNN
from models.resnet import ResNetStack


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


# get model params in config
def model_selection(model_name):
    if model_name == 'robustcnn':
        return RobustCNN(n_class=24, softmax=False)
    elif model_name == 'resnet':
        return ResNetStack()
    else:
        raise NotImplementedError(model_name)

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


def cosine_similarity(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

def wandb_set(on_wandb=False):
    ################## Wandb setting #############################
    wandb.init(
        # set the wandb project where this run will be logged
        project="AMC_few-shot",
        # group=self.config['fs_model'],
        # group="Test Sweep",
        group="extension_test",
        name=now,
        notes=f'num_support:{self.config["num_support"]},'
            f'num_query:{self.config["num_query"]},'
            f'robust:{True},'
            f'snr_range:{self.config["snr_range"]},'
            f'train_class_indice:{self.config["train_class_indice"]},'
            f'test_class_indice:{self.config["test_class_indice"]}'
            f'train_sample_size:{self.config["train_sample_size"]},'
            f'test_sample_size:{self.config["test_sample_size"]}',

        # track hyperparameters and run metadata
        config={
            # "learning_rate": self.config["lr"],
            "architecture": self.config['fs_model'],
            "dataset": "RML2018",
            # "epochs": self.config["epoch"],

            # "learning_rate": 0.01,
            # "momentum": 0.9,
            # "batch_size": 128,
            # "epochs": 30,
            # "scheduler_step_size": 10,
            # "scheduler_gamma":0.1

        }
    )

    with open('sweep_patch.yaml') as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        
    run = wandb.init(config=sweep_config)
    patch_size = wandb.config.patch_size
    w_config = wandb.config

    return patch_size


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
