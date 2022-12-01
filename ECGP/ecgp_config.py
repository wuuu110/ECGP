#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import multiprocessing.pool
import numpy as np
import train as train


def arg_wrapper_mp(args):
    return args[0](*args[1:])

class NoDaemonProcess(mp.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NoDaemonProcessPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def eval(net, gpu_id, epoch_num, batchsize, dataset, verbose, rowSize, colSize):
    print('\tgpu_id:', gpu_id, ',', net)
    train = train.Train(dataset, validation=True, verbose=verbose, rowSize=rowSize,colSize=colSize, batchsize=batchsize)
    evaluation = train(net, gpu_id, epoch_num=epoch_num, out_model=None)
    print('\tgpu_id:', gpu_id, ', eval:', evaluation)
    return evaluation


class Evaluation(object):
    def __init__(self, gpu_num, dataset='ag_news', verbose=True, epoch_num=50, batchsize=8, rowSize=20,colSize=32):
        self.gpu_num = gpu_num
        self.epoch_num = epoch_num
        self.batchsize = batchsize
        self.dataset = dataset
        self.verbose = verbose
        self.rowSize = rowSize
        self.colSize = colSize

    def __call__(self, net_lists):
        evaluations = np.zeros(len(net_lists))
        for i in np.arange(0, len(net_lists), self.gpu_num):
            process_num = np.min((i + self.gpu_num, len(net_lists))) - i
            pool = NoDaemonProcessPool(process_num)
            arg_data = [(eval, net_lists[i+j], j+2, self.epoch_num, self.batchsize, self.dataset, self.verbose, self.rowSize,self.colSize) for j in range(process_num)]
            evaluations[i:i+process_num] = pool.map(arg_wrapper_mp, arg_data)
            pool.terminate()

        return evaluations


# network configurations
#
class ECGPInfo(object):
    def __init__(self, rows=10, cols=40, level_back=10, min_active_num=8, max_active_num=50):
        self.input_num = 1
        self.func_type = ['ConvBlock_32_1', 'ConvBlock_32_3', 'ConvBlock_32_5',# 0 1 2
                          'ConvBlock_16_1', 'ConvBlock_16_3', 'ConvBlock_16_5', # 3 4 5
                          'Gru_1_1', 'Gru_2_1','Gru_4_1', #6 7 8
                          'Gru_1_2', 'Gru_2_2','Gru_4_2', #  9 10 11
                          'Sum','Sum','Sum', #12 13 14
                          'Relu','Tanh','Softmax','Sigmoid', #15 16 17 18
                          'Layernorm','Batchnorm', #19 20
                          'Linear_32','Linear_128','Linear_64', #21 22 23
                          'Attention_4','Attention_8','Attention_16', #24 25 26
                          'Attention_4','Attention_8','Attention_16' #27 28 29
                          ]
        self.func_in_num = [1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            2, 2, 2,
                            1, 1, 1, 1,
                            1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1
                            ]
        self.out_num = 1
        self.out_type = ['full']
        self.out_in_num = [1]

        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.level_back = level_back
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num
        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max([np.max(self.func_in_num), np.max(self.out_in_num)])
