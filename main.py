import logging
import argparse
from runner.train import Trainer
from runner.test import Tester
from runner.utils import CustomFormatter
from datetime import datetime

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('lr_mode', type=str, default='ml', help='Select learning method: sl(supervised-learing), ml(meta-learning)')
    parser.add_argument('mode', type=str, default='all', help='train: only train, test: only test, all: train+test')

    args = parser.parse_args()

    cfg_path = './config/config.yaml'
    model_cfg_path = './config/model_params.yaml'

    # Supervised-Learning
    if args.lr_mode == 'sl':
        if args.mode in ['train', 'all']:
            logger.info('Start Supervised-Learning')
            trainer = Trainer(cfg_path, model_cfg_path)
            trainer.train()
        if args.mode in ['test', 'all']:
            tester = Tester(cfg_path, model_cfg_path, per_snr=True)
            tester.test()

    # Meta-Learning
    elif args.lr_mode == 'ml':
        if args.mode in ['train', 'all']:
            logger.info('Start Meta-Learning')
            trainer = Trainer(cfg_path, model_cfg_path)
            trainer.fs_train()

        if args.mode in ['test', 'all']:
            tester = Tester(cfg_path, model_cfg_path)
            logger.info('Test')
            tester.fs_test()

    else:
        logger.error('Wrong argument!')

