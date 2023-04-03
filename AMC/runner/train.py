import os

import torch
import torch.nn as nn
import torch.utils.data as DATA
from torch.optim import lr_scheduler
import tqdm
from runner.utils import get_config, model_selection
from data.dataset import AMCDataset


class Trainer:
    def __init__(self, config, model_path=None):
        self.config = get_config(config)
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.batch_size = self.config["batch_size"]
        self.model_path = model_path

        self.net = model_selection(self.config["model_name"])

        # optimizer
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config['lr'], momentum=0.9)

        # loss
        self.loss = nn.CrossEntropyLoss()

        # scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if self.use_cuda:
            self.net.to(self.device_ids[0])
            self.loss.to(self.device_ids[0])

    def train(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])

        if not os.path.exists(self.config["save_path"]):
            os.mkdir(self.config["save_path"])

        train_data = AMCDataset(self.config["dataset_path"], snr_range=self.config["snr_range"])
        train_dataset_size = len(train_data)

        train_dataloader = DATA.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        if self.model_path is not None:
            self.net.load_state_dict(torch.load(self.model_path))

        for epoch in range(self.config["epoch"]):
            print('Epoch {}/{}'.format(epoch + 1, self.config["epoch"]))
            print('-' * 10)

            self.net.train()

            train_loss = 0.0

            iteration = 0
            for i, sample in enumerate(tqdm.tqdm(train_dataloader)):
                if self.use_cuda:
                    x = sample["data"].to(self.device_ids[0])
                    labels = sample["label"].to(self.device_ids[0])
                    # snr = sample["snr"].to(self.device_ids[0])
                else:
                    x = sample["data"]
                    labels = sample["label"]

                self.optimizer.zero_grad()

                outputs = self.net(x)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                iter_loss = loss.data.item()
                train_loss += iter_loss
                iteration += 1
                if not (iteration % self.config['print_iter']):
                    print('iteration {} train loss: {:.8f}'.format(iteration, iter_loss / self.batch_size))

            epoch_loss = train_loss / train_dataset_size
            print('epoch train loss: {:.8f}'.format(epoch_loss))

            self.scheduler.step()

            torch.save(self.net.state_dict(), os.path.join(self.config["save_path"], "{}.tar".format(epoch)))
            print("saved at {}".format(os.path.join(self.config["save_path"], "{}.tar".format(epoch))))