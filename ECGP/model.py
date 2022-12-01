#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math
import copy
import torch.nn.functional as F
import sys
import numpy as np
from models.embedder import Embedder
from data_reader import get_train_test_loader

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock, self).__init__()
        #
        pad = int((kernel-1)/2)
        self.c1 = nn.Conv1d(in_size, out_size, kernel, padding = pad)
    def forward(self, inputs):
        inputs = inputs.transpose(1,2)
        outputs = self.c1(inputs)
        outputs = outputs.transpose(1,2)
        return outputs


class SelfAttention(nn.Module):
    def __init__(self, d_model=32, num_heads=8):
        super(SelfAttention, self).__init__()
        #print(d_model)
        #print(num_heads)
        self.attention = nn.MultiheadAttention(d_model, num_heads)
    def forward(self, inputs):
        input_seq=inputs.transpose(0,1)
        attn_output, attn_output_weights = self.attention(input_seq, input_seq, input_seq, need_weights=False)
        attn_output=attn_output.transpose(0,1)
        return attn_output

class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()
    def forward(self, input1, input2):
        if input2.shape[2] > input1.shape[2]:
            num = int(input2.shape[2] - input1.shape[2])
            input1 = F.pad(input=input1, pad=(0, num , 0, 0, 0, 0), mode='constant',value=0)
            output = input1 + input2

        elif input1.shape[2] > input2.shape[2]:
            num = input1.shape[2] - input2.shape[2]
            input2 = F.pad(input=input2, pad=(0, num, 0, 0, 0, 0), mode='constant',value=0)
            output = input1 + input2
        else:
            output = input1 + input2
        return output
class Relu(nn.Module):
    def __init__(self):
        super(Relu, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, inputs):
        outputs = self.relu(inputs)
        return outputs
class LinearFunction(nn.Module):
    def __init__(self, dim, dim2 ):
        super(LinearFunction, self).__init__()
        self.linear = nn.Linear(dim,dim2)
        self.dropout = nn.Dropout(0.1)
    def forward(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.dropout(outputs)
        return outputs
class Gru(nn.Module):
    def __init__(self,in_size,out_size,num_layers,bidirectional,device):
        super(Gru,self).__init__()
        self.num_directions=2 if bidirectional else 1
        self.num_layers=num_layers
        self.hidden_dim=out_size
        self.device = device
        self.gru=torch.nn.GRU(input_size=in_size,hidden_size=out_size,num_layers=num_layers,batch_first=True,bidirectional=bidirectional)
    def forward(self,inputs):
        self.hidden=torch.autograd.Variable(torch.zeros(self.num_directions*self.num_layers,inputs.size()[0],self.hidden_dim)).to(self.device)
        out,self.hidden=self.gru(inputs,self.hidden)
        if self.num_directions==2:
            out=out[:,:,:self.hidden_dim]+out[:,:,self.hidden_dim:self.hidden_dim*2]
        return out

class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid,self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self,inputs):
        out=self.sigmoid(inputs)
        return out

class Tanh(nn.Module):
    def __init__(self):
        super(Tanh,self).__init__()
        self.tanh = nn.Tanh()
    def forward(self,inputs):
        out=self.tanh(inputs)
        return out

class Softmax(nn.Module):
    def __init__(self):
        super(Softmax,self).__init__()
        self.softmax=nn.Softmax(dim=2)
    def forward(self,inputs):
        out=self.softmax(inputs)
        return out
class layer_norm(nn.Module):
    def __init__(self,d_model):
        super(layer_norm, self).__init__()
        self.ly = nn.LayerNorm(d_model)
    def forward(self, inputs):
        outputs = self.ly(inputs)
        return outputs
class Batch_Norm(nn.Module):
    def __init__(self,d_model):
        super(Batch_Norm,self).__init__()
        self.d_model=d_model
        self.bm=nn.BatchNorm1d(self.d_model)
    def forward(self,inputs):
        outputs=self.bm(inputs)
        return outputs
class Glove(nn.Module):
    def __init__(self,vocab_size,colSize,rowSize):
        super(Glove,self).__init__()
        self.pth ='.'
        self.data_reader = get_train_test_loader(self.pth)
        self.embedder = Embedder(vocab_size, colSize, 0.1, rowSize)
        if self.data_reader.get_vocab().shape[1]<colSize:
            pretrained_weight = np.c_[np.array(self.data_reader.get_vocab()),np.zeros((self.data_reader.get_vocab().shape[0],colSize-self.data_reader.get_vocab().shape[1]))]
        else:
            pretrained_weight = np.array(self.data_reader.get_vocab()[:,:colSize])
        self.embedder.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
    def forward(self,x):
        out=self.embedder(x)
        return out

class ECGP2DNN(nn.Module):
    def __init__(self, ecgp, n_class, rowSize, colSize,vocab_size,device):
        super(ECGP2DNN, self).__init__()
        self.ecgp = ecgp
        self.arch = OrderedDict()
        self.encode = []
        self.rowsize = [None for _ in range(500)]
        self.colsize = [None for _ in range(500)]
        self.rowsize[0] = rowSize
        self.colsize[0] = colSize
        self.device = device
        self.embedder = Glove(vocab_size, colSize, rowSize)
        # encoder
        i = 0

        #for name, in1 in self.cgp:
        for i in range(len(self.ecgp)):
            name = self.ecgp[i][0]
            in1 = self.ecgp[i][1]
            in2 = self.ecgp[i][2]

            if name == 'input' in name:
                continue
            elif name == 'full':

                self.encode.append(nn.Linear(self.colsize[in1], n_class))
            elif name == 'Sum':
                small_in_id, large_in_id = (in1, in2) if self.colsize[in1] < self.colsize[in2] else (in2, in1)
                #
                if  self.rowsize[in1] != self.rowsize[in2]:
                    print("row size error")
                self.rowsize[i] = self.rowsize[in1]
                self.colsize[i] = self.colsize[large_in_id]
                self.encode.append(Sum())
            elif name == 'Gru_1_1' or name == 'Gru_2_1' or name == 'Gru_4_1':
                key = name.split('_')
                num_layer = int(key[1])
                self.rowsize[i] = self.rowsize[in1]
                self.colsize[i] = self.colsize[in1]
                self.encode.append(Gru(self.colsize[in1], self.colsize[in1], num_layer, False,self.device))
            elif name == 'Gru_1_2' or name == 'Gru_2_2' or name == 'Gru_4_2':
                key = name.split('_')
                num_layer = int(key[1])
                self.rowsize[i] = self.rowsize[in1]
                self.colsize[i] = self.colsize[in1]
                self.encode.append(Gru(self.colsize[in1], self.colsize[in1], num_layer, True,self.device))
            elif name == 'Relu':
                self.rowsize[i] = self.rowsize[in1]
                self.colsize[i] = self.colsize[in1]
                self.encode.append(Relu())
            elif name == 'Softmax':
                self.rowsize[i] = self.rowsize[in1]
                self.colsize[i] = self.colsize[in1]
                self.encode.append(Softmax())
            elif name == 'Tanh':
                self.rowsize[i] = self.rowsize[in1]
                self.colsize[i] = self.colsize[in1]
                self.encode.append(Tanh())
            elif name == 'Sigmoid':
                self.rowsize[i] = self.rowsize[in1]
                self.colsize[i] = self.colsize[in1]
                self.encode.append(Sigmoid())
            elif name == 'Linear_32' or name == 'Linear_64' or name == 'Linear_128':
                key = name.split('_')
                out_size = int(key[1])
                self.rowsize[i] = self.rowsize[in1]
                self.colsize[i] = out_size
                self.encode.append(LinearFunction(self.colsize[in1], out_size))
            elif name == 'Layernorm':
                self.rowsize[i] = self.rowsize[in1]
                self.colsize[i] = self.colsize[in1]
                self.encode.append(layer_norm(self.colsize[i]))
            elif name == 'Batchnorm':
                self.rowsize[i] = self.rowsize[in1]
                self.colsize[i] = self.colsize[in1]
                self.encode.append(Batch_Norm(self.rowsize[i]))
            elif name == 'Attention_4' or name == 'Attention_8' or name == 'Attention_16':
                key = name.split('_')
                head = int(key[1])
                self.rowsize[i]=self.rowsize[in1]
                self.colsize[i] = self.colsize[in1]
                self.encode.append(SelfAttention(self.colsize[i], head))
            elif name == 'ConvBlock_32_1' or name == 'ConvBlock_32_3' or name == 'ConvBlock_32_5':
                key = name.split('_')
                out_size = int(key[1])
                kernel = int(key[2])
                self.rowsize[i] = self.rowsize[in1]
                self.colsize[i] = out_size
                self.encode.append(ConvBlock(self.colsize[in1], out_size, kernel))
            elif name == 'ConvBlock_16_1' or name == 'ConvBlock_16_3' or name == 'ConvBlock_16_5':
                key = name.split('_')
                out_size = int(key[1])
                kernel = int(key[2])
                self.rowsize[i] = self.rowsize[in1]
                self.colsize[i] = out_size
                self.encode.append(ConvBlock(self.colsize[in1], out_size, kernel))
            else:
                print("init error")
            #i += 1


        #
        self.layer_module = nn.ModuleList(self.encode)
        self.outputs = [None for _ in range(len(self.ecgp))]

    def main(self,x):
        #
        outputs = self.outputs
        outputs[0] = x    # input image
        nodeID = 1
        for layer in self.layer_module:
            if isinstance(layer, ConvBlock):
                outputs[nodeID] = layer(outputs[self.ecgp[nodeID][1]])
            elif isinstance(layer,Relu):
                outputs[nodeID] = layer(outputs[self.ecgp[nodeID][1]])
            elif isinstance(layer,Softmax):
                outputs[nodeID] = layer(outputs[self.ecgp[nodeID][1]])
            elif isinstance(layer,Tanh):
                outputs[nodeID] = layer(outputs[self.ecgp[nodeID][1]])
            elif isinstance(layer,Sigmoid):
                outputs[nodeID] = layer(outputs[self.ecgp[nodeID][1]])
            elif isinstance(layer, LinearFunction):
                outputs[nodeID] = layer(outputs[self.ecgp[nodeID][1]])
            elif isinstance(layer, layer_norm):
                outputs[nodeID] = layer(outputs[self.ecgp[nodeID][1]])
            elif isinstance(layer, Batch_Norm):
                outputs[nodeID] = layer(outputs[self.ecgp[nodeID][1]])
            elif isinstance(layer, Gru):
                outputs[nodeID] = layer(outputs[self.ecgp[nodeID][1]])
            elif isinstance(layer, torch.nn.modules.linear.Linear):
                outputs[nodeID] = layer(outputs[self.ecgp[nodeID][1]])
            elif isinstance(layer, Glu):
                outputs[nodeID] = layer(outputs[self.ecgp[nodeID][1]])
            elif isinstance(layer, Sum):
                outputs[nodeID] = layer(outputs[self.ecgp[nodeID][1]], outputs[self.ecgp[nodeID][2]])
            elif isinstance(layer, SelfAttention):
                outputs[nodeID] = layer(outputs[self.ecgp[nodeID][1]])
            else:
                sys.exit("Error at model forward")
            nodeID += 1
        return outputs[nodeID-1]

    def forward(self, x):
        x = self.embedder(x)
        x = self.main(x)
        x = x.max(dim=1)[0]
        return F.log_softmax(x, dim=1)
