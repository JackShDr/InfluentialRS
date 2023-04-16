#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   evaluator.py
@Time    :   2023/03/02 19:14:36
@Author  :   Haoren Zhu 
@Contact :   hzhual@connect.ust.hk
'''
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import PositionalEncoding

class SampleNet(nn.Module):
    """
    A sample user-oriented RS used by the evaluator. 
    """
    
    def __init__(self, config):
        super(SampleNet, self).__init__()
        
        # Task configurations
        self.PAD_ID = 0
        self.item_embed_path = None
        self.n_item = config.n_item
 
        # Hyperparameters
        self.max_len = config.max_len
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.embed_dim = config.emb_dim
        self.ffn_dim = config.ffn_dim
        self.dropout = config.dropout
        
        # Network components
        self.word_embedder = nn.Embedding(self.n_item + 1, self.embed_dim,padding_idx=self.PAD_ID) 
        self.pos_embedder = PositionalEncoding(self.embed_dim, self.max_len)
        self.decoder = nn.TransformerDecoder(decoder_layer= nn.TransformerDecoderLayer(
               d_model=self.embed_dim, nhead=self.n_heads, dim_feedforward=self.ffn_dim, 
               dropout=self.dropout, activation='relu'), num_layers=self.n_layers)
        self.project = nn.Linear(self.embed_dim, self.n_item) 

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def decoding(self, dec_inp_seq):
        dec_padding_mask = dec_inp_seq.eq(self.PAD_ID)
        enc_output = torch.zeros(self.max_len,dec_inp_seq.size(0),self.embed_dim).to(dec_inp_seq.device)
        dec_inp = self.word_embedder(dec_inp_seq) * math.sqrt(self.embed_dim) + self.pos_embedder(dec_inp_seq)
        dec_inp = F.dropout(dec_inp, self.dropout, self.training)
        dec_inp = dec_inp.transpose(0, 1)
        tgt_subseq_mask = self._generate_square_subsequent_mask(dec_inp_seq.size(1)).to(dec_inp_seq.device)
        dec_output = self.decoder(
            tgt=dec_inp, memory=enc_output, tgt_mask=tgt_subseq_mask,
            tgt_key_padding_mask=dec_padding_mask
        )  # [seq_len x batch_size x embed_dim]
        dec_output = dec_output.transpose(0, 1)
        return dec_output

    def forward(self, dec_inp_seq):
        output = self.decoding(dec_inp_seq) # [batch_size x seq_len x embed_dim]
        output = self.project(output)  # [batch_size x seq_len x n_tokens]
        return output

    
    