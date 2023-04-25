import os
import torch
import torch.utils.data as DATA
import torch.nn.functional as F
import tqdm
import wandb
from runner.utils import get_config, model_selection
from data.dataset import AMCTestDataset, FewShotDataset, FewShotDatasetForOnce
from models.proto import load_protonet_conv, load_protonet_robustcnn
from plot.conf_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt


class Tester:
    def __init__(self, config, model_path=None, save_path=None, per_snr=False):
        self.config = get_config(config)
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.batch_size = self.config['batch_size']
        self.trained_snr_range = self.config['snr_range']
        self.per_snr = per_snr

        if model_path is None:
            self.model_path = self.config['load_test_path']
            self.model_num = self.config['load_model_name']
        else:
            self.model_path = model_path

        self.net = model_selection(self.config["model_name"])

        if self.use_cuda:
            self.net.to(self.device_ids[0])

    def snr_test(self, now):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])
        n_way = len(self.config['difficult_class_indice'])
        snr_range = range(self.config["test_snr_range"][0], self.config["test_snr_range"][1] + 1, 2)

        trained_snr_range = self.trained_snr_range

        acc_per_train = []
        for t_snr in trained_snr_range:
            model_name = self.config['fs_model']

            if model_name == 'rewis':
                model = load_protonet_conv(
                    x_dim=(1, 512, 256),
                    hid_dim=32,
                    z_dim=11,
                )
            elif model_name == 'robustcnn':
                model = load_protonet_robustcnn()
            m_path = os.path.join(self.model_path, str(t_snr), self.model_num)
            model.load_state_dict(torch.load(m_path))

            f = open(os.path.join(os.path.dirname(m_path), "acc.txt"), "w")
            acc_per_snr = []

            for snr in snr_range:
                test_data = FewShotDataset(self.config["dataset_path"],
                                           num_support=self.config["num_support"],
                                           num_query=self.config["num_query"],
                                           robust=True, mode='test',
                                           snr_range=[snr, snr], divide=True)
                test_dataloader = DATA.DataLoader(test_data, batch_size=1, shuffle=True)


                conf_mat = torch.zeros(n_way, n_way)
                running_loss = 0.0
                running_acc = 0.0

                model.eval()
                with torch.no_grad():
                    for episode, sample in enumerate(tqdm.tqdm(test_dataloader)):
                        output = model.proto_test(sample)

                        a = output['y_hat'].cpu().int()
                        for cls in range(n_way):
                            conf_mat[cls, :] = conf_mat[cls, :] + torch.bincount(a[cls, :], minlength=n_way)

                        running_acc += output['acc']

                avg_acc = running_acc / (episode + 1)
                acc_per_snr.append(avg_acc)

                f.write(f"SNR {snr} Accuracy: {avg_acc}\n")
            f.close()
            acc_per_train.append(acc_per_snr)

        plt.rcParams['font.family'] = 'Arial'
        title_fontsize = 32
        xlabel_fontsize = 30
        ylabel_fontsize = 30
        xticks_fontsize = 28
        yticks_fontsize = 28

        markers = ['*', '>', 'x', '.', '^', '<']

        for i, t_snr in enumerate(trained_snr_range):
            plt.plot(snr_range, acc_per_train[i], label=f'SNR={str(t_snr)}', marker=markers[i], markersize=16)

        plt.xlabel("Signal to Noise Ratio", fontsize=xlabel_fontsize)
        plt.ylabel("Classification Accuracy", fontsize=ylabel_fontsize)
        plt.title("Classification Accuracy on RadioML 2018.01 Alpha", fontsize=title_fontsize)
        plt.xticks(fontsize=xticks_fontsize)
        plt.yticks(fontsize=yticks_fontsize)
        plt.legend(loc='lower right', framealpha=1, fontsize=xlabel_fontsize)
        plt.show()