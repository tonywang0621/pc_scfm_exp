# -*- coding: utf-8 -*-

#============================================================
#
#  Deep Learning BLW Filtering
#  Main
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================

import _pickle as pickle
from datetime import datetime
import time
import numpy as np

from utils.metrics import MAD, SSD, PRD, COS_SIM
from utils import visualization as vs

from pipeline import train_dl, test_dl
import argparse
from os.path import isfile
import yaml
from torch.utils.tensorboard import SummaryWriter
from os import makedirs
import torch
import random


random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MECGE for ECG")
    parser.add_argument("--config", type=str, default="config/MECGE.yaml") #
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--n_type', type=str, default='bw', help='noise type') # 'bw' or 'em' or 'ma' or 'all' 
    parser.add_argument('--test', action='store_true') # 'bw' or 'em' or 'ma' or 'all' 
    args = parser.parse_args()

    makedirs('results', exist_ok=True)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        config_name = args.config.split('/')[-1].split('.')[0]

    noise_versions = [1, 2]
    n_type = args.n_type
    for nv in noise_versions:
        # Data_Preparation() function assumes that QT database and Noise Stress Test Database are uncompresed
        # inside a folder called data
        
        Dataset_file = f'data/dataset_{n_type}_nv{nv}.pkl'
        with open(Dataset_file, 'rb') as input:
            Dataset = pickle.load(input)


        if not args.test:
            log_path = f'logs/{config_name}_{n_type}_nv{nv}'
            tb_writer = SummaryWriter(log_path)
            train_dl(Dataset, config_name, n_type, config, nv, tb_writer)
            tb_writer.close() 
            
        [X_test, y_test, y_pred] = test_dl(Dataset, config_name, n_type, config, nv, args.device)
        test_results = [X_test, y_test, y_pred]
        # Save Results
        with open(f'results/{config_name}_{n_type}_nv{nv}.pkl', 'wb') as output: 
                pickle.dump(test_results, output)
        print(f'Results from experiment {config_name}_{n_type}_nv{nv} saved!')



