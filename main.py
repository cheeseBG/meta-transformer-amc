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
        if args.mode in ['train', 'all']:
            logger.info('Start few-shot learning')
            trainer = Trainer("config.yaml")
            trainer.fs_train(now)

        if args.mode in ['test', 'all']:
            tester = Tester("config.yaml")
            logger.info('Original Test')
            tester.fs_test(now)
            # logger.info('New Metric Test ')
            # tester.fs_test_once(now)
    else:
        logger.error('Wrong argument!')

