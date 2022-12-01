#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import random
import torchtext
import torchtext.datasets
from torch.utils.data import DataLoader
import torch
import utils
from data_reader import get_train_test_loader
from models.embedder import Embedder
logger=utils.get_logger()
from model import ECGP2DNN
import torch.nn.functional as F

def generate_batch(batch,MAX_SEQ_LEN,data_reader):
    label = torch.tensor([data_reader.get_label_id(entry.Label) for entry in batch])
    text = [data_reader.get_token_tensor(entry.Text,MAX_SEQ_LEN) for entry in batch]
    text = [torch.unsqueeze(entry,0) for entry in text]
    text = torch.cat(text)
    return text, label



def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
class Train():
    def __init__(self, dataset_name, validation=True, verbose=True, rowSize=20,colSize=32, batchsize=8):
        self.verbose = verbose
        self.rowSize = rowSize
        self.colSize = colSize
        self.validation = validation
        self.batchsize = batchsize
        self.dataset_name = dataset_name

        # load dataset
        if dataset_name == 'sst2':
            if dataset_name == 'sst2':


                self.EMBED_DIM=colSize
                self.MAX_SEQ_LEN=rowSize
                import os
                #logger.info('Loading Data')

                if self.validation:                                                                                 
                    self.data_reader = get_train_test_loader('.')
                else:
                    self.data_reader = get_train_test_loader('.')
                #logger.info('Data Loaded')
                self.VOCAB_SIZE = self.data_reader.get_vocab_size()
                print(self.VOCAB_SIZE)
                self.n_class = self.data_reader.get_num_classes()
                print(self.n_class)
                self.train_dataset = self.data_reader.get_training_data()
                self.test_dataset = self.data_reader.get_testing_data()
            print('train num    ', len(self.train_dataset))
        else:

            exit(1)

    def __call__(self, ecgp, gpuID, epoch_num=200, out_model='mymodel.model'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print('GPUID     :', gpuID)
            print('epoch_num :', epoch_num)
            print('batch_size:', self.batchsize)
        
        torch.backends.cudnn.benchmark = True
        model = ECGP2DNN(ecgp, self.n_class, self.rowSize, self.colSize,self.VOCAB_SIZE,device)

        model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.9)

        for epoch in range(1, epoch_num+1):
            start_time = time.time()
            if self.verbose:    #
                print('epoch', epoch)
            train_loss = 0
            train_acc = 0

            data = DataLoader(self.train_dataset, batch_size=self.batchsize, shuffle=True,collate_fn=lambda b: generate_batch(b, self.rowSize, self.data_reader))
            for _, (text, target) in enumerate(data):

                if text.size(0)!=self.batchsize :
                    continue

                #embedding = embedding.to(device)
                text = text.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                try:
                    output = model(text)
                except:
                    import traceback
                    traceback.print_exc()
                    return 0
                loss = criterion(output, target)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                train_acc += (output.argmax(1) == target).sum().item()
            scheduler.step()
            import os
            logger.info(f'\tLoss: {train_loss/len(self.train_dataset):.4f}(train)\t|\tAcc: {train_acc/len(self.train_dataset ) * 100:.2f}%(train)')
            print('Train set : Average loss: {:.4f}'.format(train_loss/len(self.train_dataset)))
            print('Train set : Average Acc : {:.4f}'.format(train_acc/len(self.train_dataset )))
            print('time ', time.time()-start_time)
            if self.validation:
                if epoch == 30:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
                if epoch == epoch_num:
                    for module in model.children():
                        module.train(False)
                    t_loss = self.__test_per_std(model, criterion)
            else:
                if epoch == 5:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 10
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
                if epoch % 10 == 0:
                    for module in model.children():
                        module.train(False)
                    t_loss = self.__test_per_std(model, criterion)
                if epoch == 250:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
                if epoch == 375:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
        # save the model
        print_model_parm_nums(model)
        state = {
            'state': model.state_dict(),
        }
        torch.save(state, './autoencoder.t7')
        torch.save(model.state_dict(), './model_%d.t7' % int(gpuID))
        return t_loss


    # For validation/test
    def __test_per_std(self, model, criterion):
        test_loss = 0
        test_acc = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = DataLoader(self.test_dataset,batch_size=self.batchsize,collate_fn=lambda b: generate_batch(b,  self.rowSize, self.data_reader))
        for _, (text, target) in enumerate(data):
            if text.size(0)!=self.batchsize :
                continue
            text = text.to(device)
            target = target.to(device)
            try:
                output = model(text)
            except:
                import traceback
                traceback.print_exc()
                return 0.
            loss = criterion(output, target)
            test_loss += loss.item()
            test_acc += (output.argmax(1) == target).sum().item()
        logger.info(f'\tLoss: {test_loss/len(self.test_dataset):.4f}(valid)\t|\tAcc: {test_acc/len(self.test_dataset) * 100:.2f}%(valid)')
        print('Test set : Average loss: {:.4f}'.format(test_loss/len(self.test_dataset)))
        print('Test set : Average Acc : {:.4f}'.format(test_acc/len(self.test_dataset)))


        return (test_acc/len(self.test_dataset))
