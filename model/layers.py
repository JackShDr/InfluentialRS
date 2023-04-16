#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   layers.py
@Time    :   2023/03/02 19:10:21
@Author  :   Haoren Zhu 
@Contact :   hzhual@connect.ust.hk
'''
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Generate the positional encoding for each sequence item
    """
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1 x max_len x d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
def get_end_index(seq, pad = 0):
    """
    Obtain the end position of sequence, given post-padding.
    """
    pos = np.where(seq == pad)[0]
    if len(pos) == 0:
        return len(seq) - 1
    else:
        return pos[0] - 1
    
def get_start_index(seq, pad = 0):
    """
    Obtain the start position of sequence, given pre-padding. 
    The sequence cannot be all padding.
    """
    return np.where(seq != pad)[0][0] 

def get_item_index(seq, item):
    """
    Obtain the position of an item in a sequence.
    """
    pos = np.where(seq == item)[0]
    if len(pos) == 0:
        return -1
    else:
        return pos[0]


