import os, torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import config

class TrainSet(Dataset):
    def __init__(self):
        files = os.listdir(config.data_dir)
        self.datax = None
        self.datay = None

        for f in files:
            df = pd.read_csv(os.path.join(config.data_dir, f))
            total_len = len(df)
            div_num = total_len // 64
            label = f.split('_')[0]
            label_idx = config.labels.index(label)  # convert category to num
            labels = [label_idx for _ in range(div_num)]

            if self.datay is None:
                self.datay = labels
            else:
                self.datay += labels

            for i in range(div_num):
                div = df.iloc[i*64:(i+1) * 64, :]
                if self.datax is None:
                    self.datax = np.array([div])
                else:
                    self.datax = np.vstack([self.datax, [np.array(div)]])

        # load data from csv files
        self.datax = torch.from_numpy(self.datax).float()
        self.len = self.datax.shape[0]

    def __getitem__(self, index):
        item = self.datax[index]
        label = self.datay[index]

        return item, label

    def __len__(self):
        return self.len


class TestSet():
    # DATA SHAPE: [n_class, n_feature, n_timestamp]
    # first k-shot for support set, another for query
    def __init__(self, n_way, k_shot, k_query, win_size):
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.win_size = win_size

        files = os.listdir(config.data_dir)
        self.datax = None
        self.datay = None

        for f in files:
            df = pd.read_csv(os.path.join(config.data_dir, f))
            total_len = len(df)
            div_num = total_len // self.win_size
            label = f.split('_')[0]
            label_idx = config.labels.index(label)  # convert category to num
            labels = [label_idx for _ in range(div_num)]

            if self.datay is None:
                self.datay = labels
            else:
                self.datay += labels

            for i in range(div_num):
                div = df.iloc[i*64:(i+1) * 64, :]
                if self.datax is None:
                    self.datax = np.array([div])
                else:
                    self.datax = np.vstack([self.datax, [np.array(div)]])

        # load data from csv files
        self.datay = np.array(self.datay)
        self.len = self.datax.shape[0]

    def load_test_set(self):
        # Select supportset index randomly (each label)
        # take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        qrysz = self.k_query * self.n_way

        # return values
        x_spt, y_spt, x_qry, y_qry = None, [], None, []

        for lb in range(0, len(config.labels)):
            label_indice = np.where(self.datay == lb)[0]
            print('Label{}: {}'.format(lb, len(label_indice)))
            shot_indice = np.random.choice(label_indice, size=self.k_shot, replace=False)
            query_indice = np.random.choice(label_indice[label_indice != shot_indice],
                                            size=self.k_query, replace=False)

            if x_spt is None:
                x_spt = self.datax[shot_indice]
            else:
                x_spt = np.vstack([x_spt, self.datax[shot_indice]])
            if x_qry is None:
                x_qry = self.datax[query_indice]
            else:
                x_qry = np.vstack([x_qry, self.datax[query_indice]])
            y_spt.extend(self.datay[shot_indice])
            y_qry.extend(self.datay[query_indice])

        # [setsz, c*h, n_time] ==> [n_time, setsz, c*h]
        x_spt = np.transpose(x_spt, (2, 0, 1))
        x_qry = np.transpose(x_qry, (2, 0, 1))

        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt.astype(np.float32)), torch.from_numpy(np.array(y_spt)).long(), \
                                     torch.from_numpy(x_qry.astype(np.float32)), torch.from_numpy(np.array(y_qry)).long()

        return x_spt, y_spt, x_qry, y_qry

    def __getitem__(self, index):
        item = self.datax[index]
        label = self.datay[index]

        return item, label

    def __len__(self):
        return self.len

