import os

import torch
import torch.utils.data as DATA
import torch.nn.functional as F
from argparse import ArgumentParser
import tqdm
import numpy as np
from runner.utils import get_config, model_selection
from data.dataset import AMCTestDataset
# from utils import extract_test_sample


class Tester:
    def __init__(self, config, model_path=None, save_path=None, per_snr=False):
        self.config = get_config(config)
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.batch_size = self.config['batch_size']
        self.per_snr = per_snr

        if model_path is None:
            self.model_path = self.config['load_test_path']
        else:
            self.model_path = model_path

        self.net = model_selection(self.config["model_name"])

        if self.use_cuda:
            self.net.to(self.device_ids[0])

    def test(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])

        self.net.load_state_dict(torch.load(self.model_path))

        if not self.per_snr:
            test_data = AMCTestDataset(self.config["test_dataset_path"], robust=True, mode='easy', snr_range=self.config["snr_range"])
            test_dataloader = DATA.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

            correct = 0
            total = 0

            self.net.eval()
            with torch.no_grad():
                for i, sample in enumerate(tqdm.tqdm(test_dataloader)):
                    if self.use_cuda:
                        x = sample["data"].to(self.device_ids[0])
                        labels = sample["label"].to(self.device_ids[0])
                        # snr = sample["snr"].to(self.device_ids[0])
                    else:
                        x = sample["data"]
                        labels = sample["label"]
                    outputs = self.net(x)
                    outputs = F.softmax(outputs, dim=1)

                    _, pred = torch.max(outputs, 1)

                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

            print(f'Accuracy: : {correct / total} %')

            f = open(os.path.join(os.path.dirname(self.model_path), "acc.txt"), "w")
            f.write(f"Total Accuracy: {correct / total}\n")
            f.close()

        else:
            correct = 0
            total = 0
            snr_range = range(self.config["snr_range"][0], self.config["snr_range"][1] + 1, 2)

            f = open(os.path.join(os.path.dirname(self.model_path), "acc.txt"), "w")

            for snr in snr_range:
                test_data = AMCTestDataset(self.config["test_dataset_path"], robust=True, mode='easy', snr_range=(snr, snr))
                test_dataloader = DATA.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

                self.net.eval()
                with torch.no_grad():
                    for i, sample in enumerate(tqdm.tqdm(test_dataloader)):
                        if self.use_cuda:
                            x = sample["data"].to(self.device_ids[0])
                            labels = sample["label"].to(self.device_ids[0])
                            # snr = sample["snr"].to(self.device_ids[0])
                        else:
                            x = sample["data"]
                            labels = sample["label"]
                        outputs = self.net(x)
                        outputs = F.softmax(outputs, dim=1)

                        _, pred = torch.max(outputs, 1)

                        total += labels.size(0)
                        correct += (pred == labels).sum().item()

                print(f'Accuracy: : {correct / total}')

                f.write(f"SNR {snr} Accuracy: {correct / total}\n")
            f.close()

    def fs_test(self):
        """
        Tests the protonet
        Args:
            model: trained model
            test_x (np.array): dataloader dataframes of testing set
            test_y (np.array): labels of testing set
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
            test_episode (int): number of episodes to test on
        """
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])

        conf_mat = torch.zeros(n_way, n_way)
        running_loss = 0.0
        running_acc = 0.0

        '''
        Modified
        # Extract sample just once
        '''
        sample = extract_test_sample(n_way, n_support, n_query, test_x, test_y)
        query_samples = sample['query']

        # Create target domain Prototype Network with support set(target domain)
        z_proto = model.create_protoNet(sample)

        total_count = 0
        model.eval()
        with torch.no_grad():
            for label, sample in enumerate(tqdm.tqdm(query_samples)):
                for i in range(0, len(sample) // n_way):
                    if self.use_cuda:
                        x = sample.to(self.device_ids[0])
                    else:
                        x = sample

                    output = model.proto_test(x[i * n_way:(i + 1) * n_way], z_proto, n_way, label)
                    a = output['y_hat'].cpu().int()
                    for cls in range(n_way):
                        conf_mat[cls, :] = conf_mat[cls, :] + torch.bincount(a[cls, :], minlength=n_way)

                    running_acc += output['acc']
                    total_count += 1

        avg_acc = running_acc / total_count
        print('Test results -- Acc: {:.4f}'.format(avg_acc))
        return (conf_mat / (test_episode * n_query), avg_acc)

