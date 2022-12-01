#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
#import pandas as pd

from ecgp import *
from ecgp_config import *
from train import train


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Evolving CAE structures')
    parser.add_argument('--gpu_num', '-g', type=int, default=1, help='Num. of GPUs')
    parser.add_argument('--lam', '-l', type=int, default=4, help='Num. of offsprings')
    parser.add_argument('--net_info_file', default='network_info.pickle', help='Network information file name')
    parser.add_argument('--log_file', default='./log_cgp.txt', help='Log file name')
    parser.add_argument('--mode', '-m', default='evolution', help='Mode (evolution / retrain / reevolution)')
    parser.add_argument('--swarmsize', '-s',default=5)
    parser.add_argument('--init', '-i', action='store_true')
    args = parser.parse_args()

    if args.mode == 'evolution':
        network_info = ECGPInfo(rows=3, cols=10, level_back=3, min_active_num=10, max_active_num=60)
        with open(args.net_info_file, mode='wb') as f:

            pickle.dump(network_info, f)
        rowSize = 50
        colSize = 64
        eval_f = Evaluation(gpu_num=args.gpu_num, dataset='ag_news', verbose=True, epoch_num=20, batchsize=16, rowSize=rowSize, colSize=colSize)

        # Execute evolution
        ecgp = ECGP(network_info, eval_f, lam=args.lam, rowSize=rowSize, colSize=colSize, init=args.init)

        ecgp.modified_evolution(max_eval=100, mutation_rate=0.1, log_file=args.log_file)
    # --- Retraining evolved architecture ---

    elif args.mode == 'retrain':
        print('Retrain')
        with open(args.net_info_file, mode='rb') as f:
            network_info = pickle.load(f)
        # Load network architecture
        ecgp = ECGP(network_info, None)
        #
        data = pd.read_csv(args.log_file, header=None)  # Load log file
        ecgp.load_log(list(data.tail(1).values.flatten().astype(int)))  # Read the log at final generation
        print(ecgp._log_data(net_info_type='active_only', start_time=0))
        # Retraining the network
        temp = train('ag_news', validation=False, verbose=True, batchsize=8)
        acc = temp(ecgp.pop[0].active_net_list(), 0, epoch_num=500, out_model='retrained_net.model')
        print(acc)


    elif args.mode == 'reevolution':
        # restart evolution
        print('Restart Evolution')
        rowSize = 20
        colSize = 32 
        with open('network_info.pickle', mode='rb') as f:
            network_info = pickle.load(f)
        eval_f = Evaluation(gpu_num=args.gpu_num, dataset='ag_news', verbose=True, epoch_num=50, batchsize=8, rowSize=rowSize, colSize=colSize)
        ecgp = ECGP(network_info, eval_f, lam=args.lam, rowSize=rowSize, colSize=colSize)
        #
        data = pd.read_csv('./log_ecgp.txt', header=None)
        ecgp.load_log(list(data.tail(1).values.flatten().astype(int)))
        ecgp.modified_evolution(max_eval=250, mutation_rate=0.1, log_file='./log_restat.txt')
    elif args.mode == 'fixed':
        print('fixed frame')
        temp = train('ag_news', validation=True, verbose=True, batchsize=16,rowSize=50,colSize=64)
        ecgp = [['input', 0, 0], ['Sum', 0, 0], ['Gru_1_1', 1, 1], ['Linear_128', 1, 1], ['Linear_64', 3, 1], ['Batchnorm', 2, 1], ['Gru_2_2', 5, 3], ['Sigmoid', 4, 5], ['Gru_4_1', 7, 5], ['Sum', 6, 8], ['full', 9, 9]]
        acc = temp(ecgp, 0, epoch_num=50, out_model='retrained_net.model')
    else:
        print('Undefined mode. Please check the "-m evolution or retrain or reevolution" ')
