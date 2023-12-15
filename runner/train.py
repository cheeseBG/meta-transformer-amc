import os
import torch
import torch.nn as nn
import torch.utils.data as DATA
import torch.nn.functional as F
import tqdm
import yaml
from datetime import datetime
from torch.optim import lr_scheduler, Adam
from runner.utils import get_config, model_selection, torch_seed
from data.dataset import AMCTrainDataset, FewShotDataset
from models.proto import *


class Trainer:
    def __init__(self, config, model_config, model_path=None):
        self.config = get_config(config)
        self.model_params = get_config(model_config)[self.config['model']]
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.batch_size = self.model_params["batch_size"]
        self.model_path = model_path
        self.net = model_selection(self.config["model"])

        # If variable 'robust' is True, extend frame length to 4 x 1024
        self.robust = True if self.config['model'] == 'robustcnn' else False 

        if self.config["model"] == 'robustcnn':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.model_params['lr'], momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.model_params['lr'])

        self.loss = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=self.model_params['lr_gamma'])

        if self.use_cuda:
            self.net.to(self.device_ids[0])
            self.loss.to(self.device_ids[0])

    '''
    Supervised Learning
    '''
    def train(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])
        print(f"Model: {self.config['model']}")

        save_path = os.path.join(self.config["save_path"], self.config['model'])
        os.makedirs(save_path, exist_ok=True)

        train_data = AMCTrainDataset(self.config, robust=self.robust)
        train_dataset_size = len(train_data)
        train_dataloader = DATA.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        if self.model_path is not None:
            self.net.load_state_dict(torch.load(self.model_path))

        for epoch in range(self.model_params["epoch"]):
            print('Epoch {}/{}'.format(epoch + 1, self.model_params["epoch"]))
            print('-' * 10)

            self.net.train()

            train_loss = 0.0
            correct = 0
            total = 0
            iteration = 0

            for i, sample in enumerate(tqdm.tqdm(train_dataloader)):
                if self.use_cuda:
                    x = sample["data"].to(self.device_ids[0])
                    labels = sample["label"].to(self.device_ids[0])
                else:
                    x = sample["data"]
                    labels = sample["label"]

                self.optimizer.zero_grad()

                outputs = self.net(x)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                outputs = F.softmax(outputs, dim=1)
                _, pred = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (pred == labels).sum().item()

                iter_loss = loss.data.item()
                train_loss += iter_loss
                iteration += 1
                if not (iteration % self.config['print_iter']):
                    print('iteration {} train loss: {:.8f}'.format(iteration, iter_loss / self.batch_size))

            epoch_loss = train_loss / train_dataset_size
            print('epoch train loss: {:.8f}'.format(epoch_loss))
            print(f'Accuracy: : {correct / total}')

            self.scheduler.step()

            torch.save(self.net.state_dict(), os.path.join(save_path, "{}.tar".format(epoch)))
            print("saved at {}".format(os.path.join(save_path, "{}.tar".format(epoch))))

    '''
    Meta-Training
    '''
    def fs_train(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])
        print(f"Model: {self.config['model']}")

        patch_size = self.model_params['patch_size'] if self.config['model'] in ['vit_main', 'vit_sub'] else None

        train_data = FewShotDataset(self.config["dataset_path"],
                                        num_support=self.config["num_support"],
                                        num_query=self.config["num_query"],
                                        robust=self.robust,
                                        snr_range=self.config['snr_range'],
                                        divide=self.config['data_divide'],  # divide by train proportion
                                        sample_len=self.config["train_sample_size"])

        train_dataloader = DATA.DataLoader(train_data, batch_size=1, shuffle=True)

        # fix torch seed
        torch_seed(0)

        if model_name == 'protonet':
            model = load_protonet_conv(
                x_dim=(1, 512, 256),
                hid_dim=32,
                z_dim=24,
                config=self.config
            )
            optimizer = Adam(model.parameters(), lr=0.001)
            scheduler = lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)

        elif model_name == 'robustcnn':
            model = load_protonet_robustcnn(self.config)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.config['lr'], momentum=0.9)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=self.config["lr_gamma"])

        elif model_name == 'vit':
            model = load_protonet_vit(patch_size, self.config)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['trans_lr'])
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        
        elif model_name == 'resnet':
            model = load_protonet_resnet()
            optimizer = Adam(model.parameters(), lr=self.config['lr'])

            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        elif model_name == 'daelstm':
            model = load_protonet_daelstm(self.config)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['trans_lr'])
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        for epoch in range(self.config["epoch"]):
            print('Epoch {}/{}'.format(epoch + 1, self.config["epoch"]))
            print('-' * 10)

            # while epoch < max_epoch and not stop:
            train_loss = 0.0
            train_acc = 0.0

            for episode, sample in enumerate(tqdm.tqdm(train_dataloader)):
                optimizer.zero_grad()
                loss, output = model.proto_train(sample)
                train_loss += output['loss']
                train_acc += output['acc']
                loss.backward()
                optimizer.step()


            epoch_loss = train_loss / (episode+1)
            epoch_acc = train_acc / (episode+1)
            print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, epoch_loss, epoch_acc))
            scheduler.step()

            os.makedirs(self.config["save_path"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(self.config["save_path"], "{}.tar".format(epoch)))
            print("saved at {}".format(os.path.join(self.config["save_path"], "{}.tar".format(epoch))))
