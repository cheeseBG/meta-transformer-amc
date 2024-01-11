import os
import torch
import torch.nn as nn
import torch.utils.data as DATA
import torch.nn.functional as F
import tqdm
from datetime import datetime
from runner.utils import model_selection, torch_seed
from data.dataset import AMCTrainDataset, FewShotDataset


class Trainer:
    def __init__(self, config, model_params, model_path=None):
        self.config = config
        self.model_params = model_params
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.batch_size = self.model_params["batch_size"]
        self.model_path = model_path
        self.save_path = os.path.join(self.config["save_path"], self.config['model'])
        self.net, self.optimizer, self.scheduler = model_selection(self.config, self.model_params)
        self.loss = nn.CrossEntropyLoss()

        # If variable 'robust' is True, extend frame length to 4 x 1024
        self.robust = True if self.config['model'] == 'robustcnn' else False 
       
        if self.use_cuda:
            self.net = self.net.to(self.device_ids[0])
            self.loss = self.loss.to(self.device_ids[0])

    '''
    Supervised Learning
    '''
    def train(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])
        print(f"Model: {self.config['model']}")

        os.makedirs(self.save_path, exist_ok=True)
        train_data = AMCTrainDataset(self.config, robust=self.robust)
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

            for sample in tqdm.tqdm(train_dataloader):
                x = sample["data"]
                labels = sample["label"]

                if self.use_cuda:
                    x = x.to(self.device_ids[0])
                    labels =labels.to(self.device_ids[0])

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

            epoch_loss = train_loss / len(train_data)
            print('epoch train loss: {:.8f}'.format(epoch_loss))
            print(f'Accuracy: : {correct / total}')

            self.scheduler.step()

            torch.save(self.net.state_dict(), os.path.join(save_path, "{}.tar".format(epoch)))
            print("saved at {}".format(os.path.join(save_path, "{}.tar".format(epoch))))

    '''
    Meta-Training
    '''
    def meta_train(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])
        print(f"Model: {self.config['model']}")

        os.makedirs(self.save_path, exist_ok=True)
        train_data = FewShotDataset(self.config,
                                    snr_range=self.config['snr_range'],
                                    sample_len=self.config["train_sample_size"])

        train_dataloader = DATA.DataLoader(train_data, batch_size=1, shuffle=True)

        # fix torch seed
        torch_seed(0)

        for epoch in range(self.model_params["epoch"]):
            print('Epoch {}/{}'.format(epoch + 1, self.model_params["epoch"]))
            print('-' * 10)

            # while epoch < max_epoch and not stop:
            train_loss = 0.0
            train_acc = 0.0

            for episode, sample in enumerate(tqdm.tqdm(train_dataloader)):
                self.optimizer.zero_grad()
                loss, output = self.net.proto_train(sample)
                train_loss += output['loss']
                train_acc += output['acc']
                loss.backward()
                self.optimizer.step()


            epoch_loss = train_loss / (episode+1)
            epoch_acc = train_acc / (episode+1)
            print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, epoch_loss, epoch_acc))
            self.scheduler.step()

            os.makedirs(self.config["save_path"], exist_ok=True)
            torch.save(self.net.state_dict(), os.path.join(self.save_path, "{}.tar".format(epoch)))
            print("saved at {}".format(os.path.join(self.save_path, "{}.tar".format(epoch))))
