import os
import argparse

def parse_args(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' %(script))
    parser.add_argument('--model', default='Transformer',help='model: Transformer')
    parser.add_argument('--stop_epoch', default=100, type=int, help='stopping epoch')
    if script == 'train':
        # TODO: #make it larger than the maximum label value in base class
        # AUGMENT
        parser.add_argument('--n_class', default=5, type=int, help='total number of classes in softmax')
        parser.add_argument('--batch_sz', default=32, help='batch size')
        parser.add_argument('--lr', default=0.0001, help='learning rate')
        parser.add_argument('--dataset_sz', default=436, help='dataset size')
        parser.add_argument('--train_ratio', default=0.7, help='ratio of samples for train')
    elif script == 'test':
        parser.add_argument('--n_way', default=5, help='n way')
        parser.add_argument('--k_shot', default=1, help='k shot')
        parser.add_argument('--k_query', default=43, help='k query')
        parser.add_argument('--win_size', default=64, help='window size')
    return parser.parse_args()


def obtain_checkpoint_dir(save_dir, model):
    checkpoint_dir = '%s/checkpoints/%s' %(save_dir, model)

    if not os.path.isdir(checkpoint_dir):
        print('making directory for checkpoint....')
        os.makedirs(checkpoint_dir)

    return checkpoint_dir
