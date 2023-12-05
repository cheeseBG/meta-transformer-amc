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
    parser.add_argument('size_test', type=bool, default=False, help='If this argu is True, test size scalability')

    args = parser.parse_args()

    config_fname = f'config_{args.data}.yaml' 

    # For wandb group-name
    now = str(datetime.now())

     # Check dataset availability
    if args.data not in ['RML2018', 'RML2016']:
        print(f'{args.data} is not available!')
        exit()
    else:
        print(f'Use {args.data} dataset...')


    # Supervised learning
    if args.lr_mode == 'sv':
        if args.mode in ['train', 'all']:
            logger.info('Start supervised learning')
            trainer = Trainer(config_fname)
            trainer.train()
        if args.mode in ['test', 'all']:
            tester = Tester(config_fname, per_snr=True)
            tester.test()

    # Few shot learning
    elif args.lr_mode == 'fs':
        if args.mode in ['train', 'all']:
            logger.info('Start few-shot learning')
            trainer = Trainer(config_fname)
            trainer.fs_train(now)

        if args.mode in ['test', 'all']:
            tester = Tester(config_fname)
            logger.info('Original Test')
            tester.fs_test(now)

    else:
        logger.error('Wrong argument!')

