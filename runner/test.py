import os
import torch
import torch.utils.data as DATA
import torch.nn.functional as F
import tqdm
import numpy as np
import pandas as pd
from runner.utils import model_selection, result2csv
from data.dataset import AMCTestDataset, FewShotDataset
from plot.plotter import plot_confusion_matrix, eval_plotter

class Tester:
    def __init__(self, config, model_params, per_snr=False):
        self.config = config
        self.model_params = model_params
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.batch_size = self.model_params["batch_size"]
        self.per_snr = per_snr
        self.model_path = os.path.join(self.config['load_test_path'], self.config['model'], self.config['load_model_name'])

        # If variable 'robust' is True, extend frame length to 4 x 1024
        self.robust = True if self.config['model'] == 'robustcnn' else False 

        self.net = model_selection(self.config, self.model_params, mode='test')
        if self.use_cuda:
            self.net = self.net.to(self.device_ids[0])

    def test(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])
        print(f"Model: {self.config['model']}")

        snr_range = range(self.config["test_snr_range"][0], self.config["test_snr_range"][1] + 1, 2)

        sample_len_list = self.config['test_sample_len']
        acc_per_size = []
        self.net.load_state_dict(torch.load(self.model_path))

        for sample_len in sample_len_list:
            acc_per_snr = []

            print(f'Size {sample_len} test start')
            for snr in snr_range:
                test_data = AMCTestDataset(self.config,
                                           robust=self.robust,
                                           snr_range=[snr, snr],
                                           sample_len=sample_len)
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

                acc = correct / total
                acc_per_snr.append(acc)

            acc_per_size.append(acc_per_snr)

        if self.config['save_result']:
            result2csv(acc_per_size, sample_len_list, os.path.join(self.config['load_test_path'], self.config['model']))
        
        if self.config['show_result']:
            eval_plotter(snr_range, acc_per_size, sample_len_list)
        

    def meta_test(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])

        snr_range = range(self.config["test_snr_range"][0], self.config["test_snr_range"][1] + 1, 2)

        sample_len_list = self.config['test_sample_len']
        train_sample_len = self.config['train_sample_len']
        acc_per_size = []
 
        self.net.load_state_dict(torch.load(self.model_path))

        for sample_len in sample_len_list:
            acc_per_snr = []

            print(f'Size {sample_len} test start')
            for snr in snr_range:
                print(f'SNR: {snr} test start')
               
                test_data = FewShotDataset(self.config, 
                                           mode='test', 
                                           snr_range=[snr,snr], 
                                           sample_len=sample_len,
                                           train_sample_len= train_sample_len)
     
                test_dataloader = DATA.DataLoader(test_data, batch_size=1, shuffle=True)

                running_acc = 0.0

                self.net.eval()
                flag = True
                with torch.no_grad():
                    for episode, sample in enumerate(tqdm.tqdm(test_dataloader)):
                        if flag is True:
                            print(f'Test support set shape: {sample[0]["support"][0].shape}')
                            print(f'Test query set shape: {sample[0]["query"][0].shape}')
                            flag = False
                        output = self.net.proto_test(sample)

                        running_acc += output['acc']

                avg_acc = running_acc / (episode + 1)
                print(f'avg accuracy: {avg_acc}')
                acc_per_snr.append(avg_acc)

            acc_per_size.append(acc_per_snr)

        if self.config['save_result']:
            result2csv(acc_per_size, sample_len_list, os.path.join(self.config['load_test_path'], self.config['model']))
        
        if self.config['show_result']:
            eval_plotter(snr_range, acc_per_size, sample_len_list)


       