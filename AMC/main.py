from runner.train import Trainer
from runner.test import Tester

if __name__ == '__main__':

    # Supervised learning
    trainer = Trainer("config.yaml")
    trainer.train()
    #tester = Tester("config.yaml", per_snr=True)
    #tester.test()

    # Few shot learning
    # trainer = Trainer("config.yaml")
    # trainer.fs_train()
    # tester = Tester("config.yaml")
    # tester.fs_test()