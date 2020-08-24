#!/usr/bin/env python
import os
import torch
import torch.utils.data as Data
import numpy as np
import random
from tqdm import tqdm

class CallDataset(Data.Dataset):
    def __init__(self, records_dir):
        self.records_dir = records_dir
        self.filenames = os.listdir(records_dir)
        self.count = len(self.filenames)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        signal = np.load(self.records_dir + '/' + fname)
        read_id = os.path.splitext(fname)[0]
        return read_id, signal


class TestBatchProvider():
    def __init__(self, dataset, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.dataiter = None
        self.read_num = 0
        self.signal_pool, self.read_pool ,self.row_num_pool= [], [], []

    def build(self):
        dataloader = Data.DataLoader(
            self.dataset, batch_size=1, shuffle=self.shuffle)
        self.dataiter = dataloader.__iter__()

    def next(self):
        if self.dataiter is None:
            self.build()
        try:
            while self.read_num < self.batch_size:
                read_id, signal = self.dataiter.next()
                signal = torch.squeeze(signal, dim=0)
                # self.row_num_pool.append(signal.shape[0])
                self.signal_pool.append(signal)
                self.read_pool.append(read_id)
                self.read_num += 1
            # whole_signal = torch.cat(self.signal_pool, dim=0)
            read_id_list = self.read_pool
            signal_batch_list=self.signal_pool
            # row_num_list = self.row_num_pool
            self.signal_pool, self.read_pool ,self.row_num_pool= [], [], []
            self.read_num = 0
            return signal_batch_list, read_id_list
        except StopIteration:
            return None, None
