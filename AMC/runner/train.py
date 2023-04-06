import os

import torch
import torch.nn as nn
import torch.utils.data as DATA
from torch.optim import lr_scheduler
import tqdm
from runner.utils import get_config, model_selection
from data.dataset import AMCTrainDataset
#from utils import extract_train_sample


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

        train_data = AMCTrainDataset(self.config["dataset_path"], robust=True, snr_range=self.config["snr_range"])
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

    # def fs_train(self, model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size):
    #     """
    #     Trains the protonet
    #     Args:
    #         model
    #         optimizer
    #         train_x (np.array): dataloader dataframes of training set
    #         train_y(np.array): labels of training set
    #         n_way (int): number of classes in a classification task
    #         n_support (int): number of labeled examples per class in the support set
    #         n_query (int): number of labeled examples per class in the query set
    #         max_epoch (int): max epochs to train on
    #         epoch_size (int): episodes per epoch
    #     """
    #     # divide the learning rate by 2 at each epoch, as suggested in paper
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    #     epoch = 0  # epochs done so far
    #     stop = False  # status to know when to stop
    #
    #     while epoch < max_epoch and not stop:
    #         running_loss = 0.0
    #         running_acc = 0.0
    #
    #         for episode in tqdm.tnrange(epoch_size, desc="Epoch {:d} train".format(epoch + 1)):
    #             sample = extract_train_sample(n_way, n_support, n_query, train_x, train_y)
    #             optimizer.zero_grad()
    #             loss, output = model.proto_train(sample)
    #             running_loss += output['loss']
    #             running_acc += output['acc']
    #             loss.backward()
    #             optimizer.step()
    #         epoch_loss = running_loss / epoch_size
    #         epoch_acc = running_acc / epoch_size
    #         print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, epoch_loss, epoch_acc))
    #
    #         epoch += 1
    #         scheduler.step()