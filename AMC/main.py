import logging
from runner.train import Trainer
from runner.test import Tester
from runner.utils import CustomFormatter
from datetime import datetime
from runner.utils import get_config
import pandas as pd
import random

if __name__ == '__main__':
    def save_result(result_list, test_num):
        tmp_dict = dict()
        for i in range(test_num):
            tmp_dict[f'test_{i+1}'] = result_list[i]
        df = pd.DataFrame(tmp_dict)
        df.to_csv('result.csv', index=False)


    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)


    # For wandb group-name
    now = str(datetime.now())

    config = get_config("config.yaml")

    test_case = config["test_case"]
    classes = config["class_indice"]
    train_check = [0] * (len(classes))

    test_list = []
    for t in range(test_case):
        train_mods = random.sample(classes, config["train_num"])
        test_mods = [mod for mod in classes if mod not in train_mods]

        for i in train_mods:
            train_check[i] += 1

        # # Supervised learning
        # if args.lr_mode == 'sv':
        #     if args.mode in ['train', 'all']:
        #         logger.info('Start supervised learning')
        #         trainer = Trainer("config.yaml")
        #         trainer.train()
        #
        #     if args.mode in ['test', 'all']:
        #         tester = Tester("config.yaml", per_snr=True)
        #         tester.test()

        # Few shot learning
        #logger.info(f'Start few-shot learning test case {t+1}')
        #trainer = Trainer("config.yaml", train_mods, test_mods)
        #trainer.fs_train(now)

        tester = Tester("config.yaml", train_mods, test_mods)
        logger.info('Size Test')
        acc_per_snr = tester.unseen_test(now)

        test_list.append(acc_per_snr)

        # Save result
        save_result(test_list, test_case)
        tmp_dict = {'train_count': train_check}
        df = pd.DataFrame(tmp_dict)
        df.to_csv('train_check.csv', index=False)
