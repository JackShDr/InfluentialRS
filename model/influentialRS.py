#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   influentialRS.py
@Time    :   2023/03/02 19:11:38
@Author  :   Haoren Zhu 
@Contact :   hzhual@connect.ust.hk
'''
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim import lr_scheduler

from model.layers import PositionalEncoding, get_item_index


class InfluentialNet(nn.Module):
    """
    Influential Neural Network
    
    The core NN structure for Influential Recommender System. 
    """

    def __init__(self, config):
        super(InfluentialNet, self).__init__()

        # Task configurations
        self.PAD_ID = 0  # The padding index
        self.item_embed_path = None  # Path of pretrained item embedding (if given)
        self.user_embed_path = None  # Path of pretrained user embedding (if given)
        self.n_item = config.n_item  # Number of items
        self.n_user = config.n_user  # Number of users
        self.use_u = False  # Whether to use user embeddings

        # Hyperparameters
        self.max_len = config.max_len  # Maximum length of user sequence (with paddings)
        self.n_layers = config.n_layers  # Number of decoder layers
        self.n_heads = config.n_heads  # Number of attention headers
        self.embed_dim = config.emb_dim  # Size of item embeddings
        self.u_embed_dim = config.u_emb_dim  # Size of user embeddings
        self.ffn_dim = config.ffn_dim  # Size of output FCN
        self.dropout = config.dropout  # Dropout rate

        # Initialize user and item embeddings
        self.item_embedder = self._init_embed(
            self.n_item + 1,  # Number of item plus 1 padding
            self.embed_dim,
            embed_path=self.item_embed_path,
            pad_idx=self.PAD_ID)

        self.user_embedder = self._init_embed(
            self.n_user,
            self.u_embed_dim,
            embed_path=self.user_embed_path,
            pad_idx=None  # User has no padding
        )

        # Initialize positional encoding
        self.pos_embedder = PositionalEncoding(self.embed_dim, self.max_len)

        # Decoder
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=self.n_heads,
                dim_feedforward=self.ffn_dim,
                dropout=self.dropout,
                activation='relu'),
            num_layers=self.n_layers)

        self.user_mask_layer = nn.Linear(self.u_embed_dim, 1)

        # Project and output layer
        if self.use_u == True:
            self.project = nn.Linear(self.embed_dim + self.u_embed_dim,
                                     self.n_item)
        else:
            self.project = nn.Linear(self.embed_dim, self.n_item)

        # Optimizer and learning rate scheduler
        self.optimizer = optim.Adam(filter(lambda x: x.requires_grad,
                                           self.parameters()),
                                    betas=(0.9, 0.98),
                                    eps=1e-09,
                                    lr=config.lr1)

        self.pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                               factor=0.5,
                                                               patience=4,
                                                               verbose=True)

    def _init_embed(self, num, embed_dim, embed_path=None, pad_idx=None):
        """
        Generate user/item embedding layer

        Args:
            num (int): number of users/items
            embed_dim (int): dimension of embeddings
            embed_path (string, optional): path of pretrained embeddings
            pad_idx (int, optional): padding index
            
        Returns:
            _type_: _description_
        """
        if embed_path is None:  # Use random generated embedding layers
            embed = nn.Embedding(num, embed_dim, padding_idx=pad_idx)
        else:  # Use pretrained embeddings
            weights = np.load(embed_path, allow_pickle=True)
            weights = torch.tensor(weights, dtype=torch.float)
            embed = nn.Embedding.from_pretrained(weights,
                                                 freeze=True,
                                                 padding_idx=pad_idx)
        return embed

    def _generate_square_subsequent_mask(self,
                                         size,
                                         w_h=0.05,
                                         w_obj=1,
                                         pi_factor=None):
        """
        Generate IRS masking scheme

        Args:
            size (int): size of mask matrix
            w_h (float, optional): weight of historical items. Defaults to 0.05.
            w_obj (int, optional): weight of objective item. Defaults to 1.
            pi_factor (optional): batch of personalized impressionability factors. Defaults to None.

        Returns:
            _type_: _description_
        """

        # Generate fixed masking scheme for input sequence
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(w_h))

        # Modify the weight of objective item
        if pi_factor is not None:  # Use personalized impressionability factor
            N = pi_factor.size(0)
            mask_3d = torch.zeros((N, size, size), dtype=float)
            for i in range(N):
                new_mask = mask.clone()
                new_mask[:, -1] = w_obj * pi_factor[i]
                mask_3d[i] = new_mask
            mask = torch.repeat_interleave(mask_3d, self.n_heads, dim=0)
        else:  # Do not use personalized impressionability factor
            mask[:, -1] = w_obj

        return mask

    def decoding(self, dec_input_seq, user, return_pi=False):
        """
        Decoding process

        Args:
            dec_input_seq (tensor): batch of viewing item index sequences
            user (tensor): batch of user index
            return_pi (bool, optional): whether to return PIF. Defaults to False.

        Returns:
            _type_: _description_
        """

        # Generate decoder inputs
        dec_padding_mask = dec_input_seq.eq(self.PAD_ID)
        enc_output = torch.zeros(self.max_len, dec_input_seq.size(0),
                                 self.embed_dim).to(dec_input_seq.device)
        dec_input = self.item_embedder(dec_input_seq) * math.sqrt(
            self.embed_dim) + self.pos_embedder(dec_input_seq)
        dec_input = F.dropout(dec_input, self.dropout, self.training)
        dec_input = dec_input.transpose(0, 1)

        # Calculate personalized impressionability factor
        pi_factor = self.user_mask_layer(self.user_embedder(user))

        # Generate masking
        tgt_subseq_mask = self._generate_square_subsequent_mask(
            dec_input_seq.size(1), pi_factor)
        tgt_subseq_mask = tgt_subseq_mask.type(torch.float32).to(
            dec_input_seq.device)

        # Generate decoder output
        dec_output = self.decoder(tgt=dec_input,
                                  memory=enc_output,
                                  tgt_mask=tgt_subseq_mask,
                                  tgt_key_padding_mask=dec_padding_mask
                                  )  # [seq_len x batch_size x embed_dim]
        dec_output = dec_output.transpose(0, 1)

        # Generate output
        if return_pi:  # If return PIF
            return dec_output, pi_factor

        return dec_output

    def forward(self, dec_input_seq, user):

        # Generate decoder output
        x = self.decoding(dec_input_seq,
                          user)  # [batch_size x seq_len x embed_dim]

        # Generate user embedding
        if self.use_u == True:
            u_vec = self.user_embedder(user)
            u_vec = u_vec.repeat(self.max_len, 1, 1).permute(1, 0, 2)
            x = torch.cat((x, u_vec), dim=2)

        x = self.project(x)  # [batch_size x seq_len x n_tokens]

        return x


class IRSNN(nn.Module):
    """
    Functionality handler for IRS.

    IRSNN uses InfluentialNet as the core network structure and implements tasks such as sequence generation.
    """

    def __init__(self, config, net, device):
        super(IRSNN, self).__init__()
        
        # Task configuration
        self.PAD_ID = 0
        self.n_item = config.n_item
        self.embed_dim = config.emb_dim
        
        # Core network components
        self.net = net
        self.device = device
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(filter(lambda x: x.requires_grad,
                                           self.net.parameters()),
                                    betas=(0.9, 0.98),
                                    eps=1e-09,
                                    lr=config.lr1)
        self.pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                               factor=0.5,
                                                               patience=4,
                                                               verbose=True)

        
        # Output probability structure
        self.softmax = nn.Softmax(dim=2)

    def get_loss_on_eval_data(self, seqs, users):
        """
        Calculate the loss using the corresponding loss function on evaluation data.

        Args:
            seqs (tensor): batch of viewing item sequences
            users (tensor): batch of user index

        Returns:
            float: loss value
        """
        self.net.eval()
        dec_seqs = seqs.clone()
        output = self.net.forward(dec_seqs, users)
        output = output[:, :-1, :].contiguous()
        output = output.view(-1, self.n_item)

        dec_tgt_seq = seqs[:, 1:].contiguous().view(-1)
        mask = dec_tgt_seq.gt(self.PAD_ID)
        masked_target = dec_tgt_seq.masked_select(mask)
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.n_item)
        masked_output = output.masked_select(output_mask).view(-1, self.n_item)
        loss = self.loss_function(masked_output, masked_target - 1)

        return loss.item()

    def train_batch(self, seqs, users):
        """
        Train the core InfluentialNet.

        Args:
            seqs (tensor): batch of viewing item sequences
            users (tensor): batch of user index

        Returns:
            _type_: _description_
        """

        # Calculate the loss function
        self.net.train()
        dec_seqs = seqs.clone()
        output = self.net.forward(dec_seqs, users)
        output = output[:, :-1, :].contiguous()
        output = output.view(-1, self.n_item)

        dec_tgt_seq = seqs[:, 1:].contiguous().view(-1)
        mask = dec_tgt_seq.gt(self.PAD_ID)
        masked_target = dec_tgt_seq.masked_select(mask)  #

        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.n_item)
        masked_output = output.masked_select(output_mask).view(-1, self.n_item)
        loss = self.loss_function(masked_output, masked_target - 1)

        # Update the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _delete_item_in_history(self, tensor, indices):
        """
        Remove items belong to the given set from the viewing item sequences

        Args:
            tensor (tensor): a viewing item seuqence
            indices (tensor): a set of item indexes to be removed from 'tensor'

        Returns:
            _type_: _description_
        """
        return tensor[~tensor.unsqueeze(1).eq(indices).any(1)]

    def get_pif_in_batch(self, seqs, users):
        """
        Generate the personalized impressionability factors for users.

        Args:
            seqs (seqs): batch of viewing item sequences
            users (_type_): batch of user indexes

        Returns:
            _type_: _description_
        """
        dec_seqs = seqs.clone()
        _, r = self.net.decoding(dec_seqs, users, return_pi=True)
        return r.detach().cpu().numpy()

    def get_accuracy_metrics_in_batch(self,
                                      raw,
                                      seqs,
                                      users,
                                      targets,
                                      labels,
                                      top_k=20,
                                      gap_len=20,
                                      use_h=True):
        """
        Get the two accuracy metrics, Hit@top_k and MRR
        
        To be improved:
        * find the end of history from the data itself rather than passing in gap_len
        
        returns:
            hit_count: int, the number of hit in the batch
            rr: list, the reverse of rank of labelse in the batch
        """
        # Obtain the output
        dec_seqs = seqs.clone()
        output = self.net.forward(
            dec_seqs,
            users)  # [batch x seq_len x n_tokens] , n_item = n_tokens
        output = output[:, :-1, :].contiguous(
        )  # [batch x (seq_len - 1) x n_tokens]
        history_end_pos = seqs.size(1) - (
            gap_len + 1) - 1  # Size of whole sequence - (path + target) - 1

        hit_count = 0
        rr = []
        for index in range(seqs.size(0)):
            hist = raw[index].to(seqs.device)
            ## Obtain the probability of all the items
            prob_dist = output[index][history_end_pos]
            _, indices = prob_dist.sort(dim=0, descending=True)
            indices = indices + 1  # Obtain the sorted item list
            if use_h == True:
                indices = self._delete_item_in_history(
                    indices, hist)  # Filter the item in the history
            label = labels[index]  # Obtain the label

            ## Calculate Hit@top_k
            top_k_items = indices[:top_k]
            if label in top_k_items:
                hit_count += 1  # Increment hit count
            if label in indices:
                reverse_rank = np.reciprocal(
                    float(((indices == label).nonzero()[0].item() + 1)))
                rr.append(reverse_rank)
        return hit_count, np.array(rr)

    def get_seq_in_batch(self,
                         seqs,
                         users,
                         targets,
                         max_path_len=20,
                         gap_len=20,
                         sample=False,
                         sample_k=3):
        """
        Generate the path
        """
        # Initialize the output
        temp_tensors = seqs.clone()  # Move the input tensors to a temp tensor
        device = temp_tensors.device
        shape = seqs.shape  # [batch_size, seq_len]
        paths = torch.zeros(
            (shape[0], max_path_len))  # batch_size x max_path_len
        history_end_pos = shape[1] - (gap_len + 1) - 1

        # Generate the path recursively
        for i in range(
                max_path_len):  # For each step in generating the sequence
            output = self.net.forward(temp_tensors, users).contiguous(
            )  # [batch x seq_len x n_tokens] , n_item = n_tokens
            # Calculate Probability (not log soft max)
            # The probability will be used to sample next item if sampling is used
            output = self.softmax(output)
            for index in range(shape[0]):
                # Find the item to be recommended
                prob, indice = output[index][history_end_pos].topk(100)
                indice = indice + 1
                history = temp_tensors[
                    index][:history_end_pos +
                           1]  # History that contains padding 0
                # Find the position of items that have occured in the history
                pos = ~indice.unsqueeze(1).eq(history).any(1)
                if sample == False:  # Select top item
                    next_item = indice[pos][0].item()
                else:  # Use top-k sampling
                    prob = prob[pos][:sample_k]
                    indice = indice[pos][:sample_k]
                    next_item = indice[torch.multinomial(
                        prob, 1, replacement=False).item()]
                # Update the next item to path and history
                paths[index][i] = next_item

                if history_end_pos < shape[
                        1] - 2:  # There are space (i.e. padding zero) before target
                    temp_tensors[index][history_end_pos + 1] = next_item
                    history_end_pos += 1  # Increment by 1
                else:
                    temp_tensor = torch.zeros(shape[1], dtype=torch.long).to(
                        device)  # Create the same shape
                    temp_tensor[:-2] = temp_tensors[index][
                        1:-1]  # Previous history
                    temp_tensor[-2] = next_item  # Next item
                    temp_tensor[-1] = temp_tensors[index][-1]  # Target
                    temp_tensors[index] = temp_tensor
                    history_end_pos = history_end_pos  # Unchange

        # Filter paths
        n_early_success = 0
        # Move to cpu
        paths = paths.detach().cpu().numpy()  # The persuasion paths
        targets = targets.detach().cpu().numpy()  # The target items
        histories = seqs[:, :-1].detach().cpu().numpy()
        actual_history = []  # The actual history
        for i in range(shape[0]):  # For each data sample
            target = targets[i]
            path = paths[i]
            history = histories[i]
            # Clean the path and count early success
            pos = get_item_index(path, target)
            if pos != -1:  # Early success
                n_early_success += 1
                path[pos + 1:] = 0  # The part after target is reset to zero

            actual_history.append(history[history != 0])
        return paths, targets, actual_history, n_early_success
