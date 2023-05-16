import os
import torch
import torch.utils.data as DATA
import torch.nn.functional as F
import tqdm
from runner.utils import get_config, model_selection
from data.dataset import AMCTestDataset, FewShotDataset, FewShotDatasetForOnce
from models.proto import load_protonet_conv, load_protonet_robustcnn, load_protonet_vit
from plot.conf_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

        snr_range = range(self.config["test_snr_range"][0], self.config["test_snr_range"][1] + 1, 2)

        sample_size_list = self.config['test_sample_size']

        acc_per_size = []

        self.net.load_state_dict(torch.load(self.model_path))

        model_name = self.config['model_name']
        robust = False
        if model_name == 'robustcnn':
            robust = True

        for sample_size in sample_size_list:
            acc_per_snr = []

            print(f'Size {sample_size} test start')
            for snr in snr_range:
                test_data = AMCTestDataset(self.config["test_dataset_path"],
                                           robust=robust,
                                           snr_range=[snr, snr],
                                           sample_len=sample_size)
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

        # Save result
        self.save_result(acc_per_size, sample_size_list, self.config["save_path"])

        # SNR Graph
        plt.rcParams['font.family'] = 'Arial'
        title_fontsize = 32
        xlabel_fontsize = 30
        ylabel_fontsize = 30
        xticks_fontsize = 28
        yticks_fontsize = 28
        legend_fontsize = 20

        markers = ['*', '>', 'x', '.', '^', '<', 'v']

        for i, sample_size in enumerate(sample_size_list):
            plt.plot(snr_range, acc_per_size[i], label=f'sample_size{str(sample_size)}', marker=markers[i],
                     markersize=16)

        plt.xlabel("Signal to Noise Ratio", fontsize=xlabel_fontsize)
        plt.ylabel("Classification Accuracy", fontsize=ylabel_fontsize)
        plt.title("Classification Accuracy on RadioML 2018.01 Alpha", fontsize=title_fontsize)
        plt.xticks(fontsize=xticks_fontsize)
        plt.yticks(fontsize=yticks_fontsize)
        plt.legend(loc='lower right', framealpha=1, fontsize=legend_fontsize)
        plt.show()

    def size_test(self, now):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])

        n_way = len(self.config['difficult_class_indice'])
        snr_range = range(self.config["test_snr_range"][0], self.config["test_snr_range"][1] + 1, 2)

        load_folder_name = self.config['save_folder_name']
        sample_size_list = self.config['test_sample_size']

        acc_per_size = []

        model_name = self.config['fs_model']
        robust = False
        if model_name != 'vit':
            robust = True

        if model_name == 'rewis':
                model = load_protonet_conv(
                    x_dim=(1, 512, 256),
                    hid_dim=32,
                    z_dim=24,
             )
        elif model_name == 'robustcnn':
            model = load_protonet_robustcnn()

        elif model_name == 'vit':
            model = load_protonet_vit()

        m_path = os.path.join(self.model_path, load_folder_name, self.config['load_model_name'])
        model.load_state_dict(torch.load(m_path))

        model_name = self.config['fs_model']
        robust = False
        if model_name == 'robustcnn':
            robust = True

        for sample_size in sample_size_list:
            acc_per_snr = []

            print(f'Size {sample_size} test start')
            for snr in snr_range:
                test_data = FewShotDataset(self.config["dataset_path"],
                                           num_support=self.config["num_support"],
                                           num_query=self.config["num_query"],
                                           robust=robust,
                                           mode='test',
                                           snr_range=[snr, snr],
                                           divide=self.config['data_divide'],
                                           sample_len=sample_size)
                test_dataloader = DATA.DataLoader(test_data, batch_size=1, shuffle=True)

                running_loss = 0.0
                running_acc = 0.0

                model.eval()
                with torch.no_grad():
                    for episode, sample in enumerate(tqdm.tqdm(test_dataloader)):
                        output = model.proto_test(sample)

                        running_acc += output['acc']

                avg_acc = running_acc / (episode + 1)
                acc_per_snr.append(avg_acc)

            acc_per_size.append(acc_per_snr)

        # Save result
        self.save_result(acc_per_size, sample_size_list, self.config["save_path"])


        # SNR Graph
        plt.rcParams['font.family'] = 'Arial'
        title_fontsize = 32
        xlabel_fontsize = 30
        ylabel_fontsize = 30
        xticks_fontsize = 28
        yticks_fontsize = 28
        legend_fontsize = 20

        markers = ['*', '>', 'x', '.', '^', '<', 'v']

        for i, sample_size in enumerate(sample_size_list):
            plt.plot(snr_range, acc_per_size[i], label=f'sample_size{str(sample_size)}', marker=markers[i],
                     markersize=16)

        plt.xlabel("Signal to Noise Ratio", fontsize=xlabel_fontsize)
        plt.ylabel("Classification Accuracy", fontsize=ylabel_fontsize)
        plt.title("Classification Accuracy on RadioML 2018.01 Alpha", fontsize=title_fontsize)
        plt.xticks(fontsize=xticks_fontsize)
        plt.yticks(fontsize=yticks_fontsize)
        plt.legend(loc='lower right', framealpha=1, fontsize=legend_fontsize)
        plt.show()


    def save_result(self, result_list, size_list, save_path):
        tmp_dict = dict()
        for i, size in enumerate(size_list):
            tmp_dict[size] = result_list[i]
        df = pd.DataFrame(tmp_dict)
        df.to_csv(os.path.join(save_path, 'result.csv'), index=False)

