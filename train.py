import os, sys, copy, random, logging, argparse, torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
import esm
from utils.dataset import SentenceDataset
from utils.data_utils import *
os.environ['CUDA_VISIBLE_DEVICES']='0'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1();
        self.tokenizer = alphabet.get_batch_converter()

        dataset = SentenceDataset(yuluo_file_names)

        train_size, test_size = int(len(dataset) * 0.8), int(len(dataset) * 0.2)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataaset=test_dataset, batch_size=opt.batch_size, shuffle=True)
        self.model = opt.model_class(opt, model).to(opt.device)
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))


    def run(self):
        criterion = nn.CrossEntropyLoss()
        max_test_acc_overall = 0
        max_f1_overall = 0
        # max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)

def main():
    model_classes = {}

    data_files = ['non-yuluo.dat', 'yuluo.dat']
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='esm_avg', type=str)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--cls_dim', default=2, type=int)

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0,type=int)
    parser.add_argument('--cuda', default='0', type=str)

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(
        opt.device)
    setup_seed(opt.seed)

    if not os.path.exists('./logs'):
        os.makedirs('./logs', mode=0o777)

    log_file = './logs/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()