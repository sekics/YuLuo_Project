import os, sys, json, pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from data_utils import read_data_from_file
class SentenceDataset(Dataset):
    def __init__(self, fnames):

        pos_data, neg_data = read_data_from_file(fnames)

        pos_count = len(pos_data)
        neg_count = len(neg_data)

        neg_data = neg_data[: pos_count * 3]

        data = pos_data.extend(neg_data)

        # all_samples = list()
        # for obj in tqdm(data, total = len(data), desc="Loading Training data and Test Data"):

        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
        