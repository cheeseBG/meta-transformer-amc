from runner.train import Trainer
from runner.test import Tester

if __name__ == '__main__':
    # trainer = Trainer("config.yaml")
    # trainer.train()
    tester = Tester("config.yaml", per_snr=True)
    tester.test()
#93.38
#0.4452 cmc full