import os, sys, json, pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from data_utils import read_data_from_file
class SequenceDataset(Dataset):
    def __init__(self, fnames, batch_converter, opt):

        pos_data, neg_data = read_data_from_file(fnames)

        