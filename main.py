import logging
import argparse
from runner.train import Trainer
from runner.test import Tester
from runner.utils import CustomFormatter, get_config
from datetime import datetime

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('mode', type=str, default='all', help='train: only train, test: only test, all: train+test')

    args = parser.parse_args()

    config = get_config('./config/config.yaml')
    model_params = get_config('./config/model_params.yaml')[config['model']]
    lr_mode = model_params['lr_mode']

    assert args.mode in ['train', 'test', 'all']

    def run_training(trainer, tester):
        if args.mode in ['train', 'all']:
            logger.info(f'Start {lr_mode.capitalize()}-Learning')
            trainer.train() if lr_mode == 'supervised' else trainer.meta_train()

        if args.mode in ['test', 'all']:
            logger.info('Test')
            tester.test() if lr_mode == 'supervised' else tester.meta_test()

    trainer = Trainer(config, model_params)
    tester = Tester(config, model_params, per_snr=(lr_mode == 'supervised'))
    run_training(trainer, tester)

