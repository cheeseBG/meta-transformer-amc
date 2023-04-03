import yaml
from models.robustcnn import RobustCNN
from models.cmcformer import ViT


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


# get model params in config
def model_selection(model_name):
    if model_name == 'robustcnn':
        return RobustCNN(n_class=24, softmax=False)
    elif model_name =="cmc":
        return ViT(input_size=(4, 1024), patch_size=(2, 8), num_classes=24, dim=128, mlp_dim=128*2, depth=5, heads=8)
    else:
        raise NotImplementedError(model_name)
