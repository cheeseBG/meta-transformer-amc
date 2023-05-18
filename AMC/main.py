import logging
from runner.train import Trainer
from runner.test import Tester
from runner.utils import CustomFormatter
from datetime import datetime
from runner.utils import get_config
import pandas as pd
import random

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)


    # For wandb group-name
    now = str(datetime.now())

    config = get_config("config.yaml")

    test_case_num = config["test_case_num"]
    classes = config["class_indice"]
    test_case = config["select_test"]
    test_classes = config[test_case]
    total_test_mods = [mod for mod in classes if mod not in test_classes]

    train_dict = {}
    test_dict = {}
    test_result_dict = {}
    for t in range(test_case_num):
        #train_mods = random.sample(classes, config["train_way"])
        test_mods = random.sample(total_test_mods, config["test_way"])

        #train_dict[f'test_{t+1}'] = train_mods
        test_dict[f'test_{t + 1}'] = test_mods

        #for i in train_mods:
        #    train_check[i] += 1

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
        #trainer = Trainer("config.yaml", train_mods, test_mods, t+1)
        #trainer.fs_train(now)

        tester = Tester("config.yaml", test_case, test_mods)
        logger.info(f'Unseen Test: {test_case}_{t}')
        acc_per_snr = tester.unseen_test(now)

        test_result_dict[f'test_{t + 1}'] = acc_per_snr

        #test_list.append(acc_per_snr)
        #print(train_dict)

    # Save result
    #save_result(test_list, t+1)
    #df = pd.DataFrame(train_dict)
    #df.to_csv('train_mods.csv', index=False)

    df = pd.DataFrame(test_dict)
    df.to_csv(f'unseen_result/{test_case}_test_mods.csv', index=False)
    df = pd.DataFrame(test_result_dict)
    df.to_csv(f'unseen_result/{test_case}_test_result.csv', index=False)
