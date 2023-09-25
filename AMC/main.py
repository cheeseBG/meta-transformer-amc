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
    parser.add_argument('lr_mode', type=str, default='fs', help='Select learning method: sv(supervised), fs(few-shot)')
    parser.add_argument('mode', type=str, default='all', help='train: only train, test: only test, all: train+test')
    parser.add_argument('data', type=str, default='RML2018', help='Two type of dataset 1.RML2018, 2.RML2016')

    args = parser.parse_args()

    # For wandb group-name
    now = str(datetime.now())

    # Supervised learning
    if args.lr_mode == 'sv':
        if args.mode in ['train', 'all']:
            logger.info('Start supervised learning')
            trainer = Trainer("config.yaml")
            trainer.train()

        if args.mode in ['test', 'all']:
            tester = Tester("config.yaml", per_snr=True)
            tester.test()

    # Few shot learning
    elif args.lr_mode == 'fs':
        if args.data not in ['RML2018', 'RML2016']:
            print(f'{args.data} is not available!')
            exit()

        if args.mode in ['train', 'all']:
            logger.info('Start few-shot learning')
            trainer = Trainer("config.yaml")
            patch_size = trainer.fs_train(now, args.data)

        if args.mode in ['test', 'all']:
            tester = Tester("config.yaml")
            logger.info('Size Test')
            tester.size_test(now, patch_size, args.data)
    else:
        logger.error('Wrong argument!')

