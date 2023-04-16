#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   evaluator.py
@Time    :   2023/03/02 19:11:34
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

from model.layers import get_end_index


class Evaluator(nn.Module):
    """
    Independent Next-item RS as an evaluator to generate probability measurement.
    """
    
    def __init__(self, config, net, device):
        """
        Create an evaluator object.

        Args:
            config (_type_): configuration parser
            net (_type_): a next-item RS
            device (_type_): cpu/gpu device to run the evaluator
        """
        super(Evaluator, self).__init__()
        
        # Task configurations
        self.PAD_ID = 0
        self.vocab_size = config.n_item

        # Core network components
        self.net = net 
        self.device = device
        self.softmax = nn.LogSoftmax(dim=2)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda x: x.requires_grad, self.net.parameters()),
            betas=(0.9, 0.98), eps=1e-09, lr = config.lr1)
        self.pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,factor=0.5,\
                                                          patience=4,verbose=True)
    
    def train_batch(self, target):
        self.net.train()
        dec_seqs = target[:, :-1]  # this does not remove the last eos ,  batch x seq_len-1
        output = self.net.forward(dec_seqs)  # [batch x (seq_len-1) x n_tokens] , vocab_size = n_tokens
        output = output.view(-1, self.vocab_size)  # [batch*(seq_len-1) x n_tokens]

        dec_tgt_seq = target[:, 1:].contiguous().view(-1)  # batch*(seq_len-1)
        mask = dec_tgt_seq.gt(self.PAD_ID)  # [(batch_sz*seq_len)]
        masked_target = dec_tgt_seq.masked_select(mask)  #
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)  # [(batch_sz*seq_len) x n_tokens]
        masked_output = output.masked_select(output_mask).view(-1, self.vocab_size)
        loss = self.loss_function(masked_output, masked_target-1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def get_loss_on_eval_data(self, eval_data):
        """
        Calculate the loss using the corresponding loss function on the evaluation data

        Args:
            eval_data (tensor): batch of viewing item sequences

        Returns:
            float: loss value
        """
        self.net.eval()
        dec_seqs = eval_data[:, :-1]  # this does not remove the last eos ,  batch x seq_len-1
        output = self.net.forward(dec_seqs)  # [batch x (seq_len-1) x n_tokens] , vocab_size = n_tokens
        output = output.view(-1, self.vocab_size)  # [batch*(seq_len-1) x n_tokens]

        dec_tgt_seq = eval_data[:, 1:].contiguous().view(-1)  # batch*(seq_len-1)
        mask = dec_tgt_seq.gt(self.PAD_ID)  # [(batch_sz*seq_len)]
        masked_target = dec_tgt_seq.masked_select(mask)  #
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)  # [(batch_sz*seq_len) x n_tokens]
        masked_output = output.masked_select(output_mask).view(-1, self.vocab_size)
        loss = self.loss_function(masked_output, masked_target - 1)
        #print('eval_loss: ', loss.item())
        return loss.item()
    
    def get_accuracy_metrics_in_batch(self, seqs, top_k = 20,  use_h=True):
        """
        Calculate Hit@top_k and MRR for the evaluator model

        Args:
            seqs (tensor): batch of viewing item sequences containing the label item
            top_k (int, optional): top_k for hit rate. Defaults to 20.
            use_h (bool, optional): whether to filter repeated item based on the viewing history. Defaults to True.

        Returns:
            tuple: [hit count, RR array]
        """
        # Obtain the output
        self.net.eval()
        dec_seqs = seqs[:, :-1]
        output = self.net.forward(dec_seqs)  # [batch x seq_len x n_tokens] , vocab_size = n_tokens
        
        hit_count = 0
        rr = []
        for index in range(seqs.size(0)):
            seq = seqs[index]
            end_pos = get_end_index(seq)
            label = seq[end_pos]
            ## Obtain the probability of all the items
            prob_dist = output[index][end_pos-1]
            
            
            _, indices = prob_dist.sort(dim=0, descending=True)
            indices = indices + 1 # Obtain the sorted item list
            if use_h == True:
                indices = self._delete_item_in_history(indices, seq[:end_pos]) # Filter the item in the history
            
            ## Calculate Hit@top_k
            top_k_items = indices[:top_k]
            if label in top_k_items:
                hit_count += 1  # Increment hit count
            if label in indices:
                reverse_rank = np.reciprocal(float(((indices==label).nonzero()[0].item()+1)))
                rr.append(reverse_rank)
        return hit_count, np.array(rr)
    
    ### Utility part for influence path evaluation
    def _get_first_none_zero_index(self, tensor):
        """
        Find the first non-zero index in the given array
        """
        zero_list = (tensor == 0).nonzero()
        if len(zero_list) == 0:
            return tensor.size()[0] - 1
        else:
            return zero_list[0].item() - 1
        
    def _get_last_path_index(self, tensor, target):
        """
        Get the endding index of the influence path
        """
        target_pos = (tensor == target).nonzero()
        if len(target_pos) == 0: # Does not exist target
            return self._get_first_none_zero_index(tensor)
        else:
            return target_pos[0].item()-1
        
    def _delete_item_in_history(self, tensor, indices):
        """
        Remove items in the indices from the tensor
        """
        return tensor[~tensor.unsqueeze(1).eq(indices).any(1)]
    
    def get_grad_in_batch(self, histories, new_seqs, targets, start_pos, l_paths):
        """
        Calculate the gradual interests along the influence path

        Args:
            histories (tensor): batch of item viewing histories
            new_seqs (_type_): batch of item sequences containing the influence path
            targets (_type_): batch of objective items
            start_pos (_type_): batch of starting indexes of the influence path
            l_paths (_type_): batch of the influence path lengths

        Returns:
            _type_: metrics related to the change of interests
        """
        t_probs = [] # Probability of accepting target item
        p_probs = [] # Probability of accepting path item
        self.net.eval()
        
        # Restore paths
        paths = []
        for i in range(new_seqs.size(0)):
            left_idx = start_pos[i]
            right_idx = left_idx + l_paths[i]
            paths.append(new_seqs[i][left_idx: right_idx].detach().cpu().numpy()) # Append the whole path to the path set
        seq_len = max(l_paths) # Calculata the maximum path length
        
        # Initialize the history tensors
        temp_tensors = histories[:, :-1]
        shape = temp_tensors.shape
        device = temp_tensors.device
        # Generate interest prob
        for i in range(seq_len): # Iterate over the maximum path length (shorter ones will be padded with zero)
            prob_dict = self.net.forward(temp_tensors)
            prob_dict = self.softmax(prob_dict) # Obtain the probability distribution on each step
            t_prob = []
            p_prob = []
            for j in range(shape[0]): # Iterate over the sequence sample
                # Calculate the prob
                end = self._get_first_none_zero_index(temp_tensors[j]) # Find the end of the sequence
                # Move forward
                if i < l_paths[j]:  # If not exceed the path length    
                    next_item = paths[j][i] # Next item in the path
                    accept_p = prob_dict[j][end][next_item-1].item() # Accept next item probability 
                    accept_t = prob_dict[j][end][targets[j]-1].item() # Accept target item probability
                else: # Exceed the path length
                    next_item = 0
                    accept_p = 0
                    accept_t = 0

                t_prob.append(accept_t) 
                p_prob.append(accept_p) 
                
                # Update the history tensor
                if end == shape[1]-1: # Full sequence
                    temp_tensor = torch.zeros(shape[1], dtype=torch.long).to(device)
                    temp_tensor[: -1] = temp_tensors[j][1:]
                    temp_tensor[-1] = next_item
                else: # Not full
                    temp_tensor = temp_tensors[j]
                    temp_tensor[end + 1] = next_item
                temp_tensors[j] = temp_tensor

            t_probs.append(t_prob)
            p_probs.append(p_prob)
            
        # Obtain the probabiltiy matrix with padding zero
        t_probs = np.array(t_probs).T # batchsize x maximum path length 
        p_probs = np.array(p_probs).T # batchsize x maximum path length
        
        # Initialize the probability list without padding
        avg_ps = [] # Average accept probabiltiy
        iois = []
        for i in range(shape[0]): # Iterate over each data sample
            t_prob = t_probs[i]
            p_prob = p_probs[i]
            target_p = t_prob[t_prob < 0] # log probability term
            path_p = p_prob[p_prob<0]
            
            iois.append(target_p[-1] - target_p[0]) # Calculate increase of interests
            avg_ps.append(sum(path_p)/len(path_p)) # Calculate accept average

        return t_probs, p_probs, avg_ps, iois
    
    def get_rr_increase_in_batch(self, histories, new_seqs, targets):
        """
        Calculate the increase of ranking/reciprocal rank)

        Args:
            histories (tensor): batch of item viewing histories
            new_seqs (tensor): batch of item sequences containing the influence path
            targets (tensor): batch of objective items

        Returns:
            _type_: metrics related to the change of rank
        """
        # Calculate history rank
        dec_seqs = histories[:,:-1].clone()
        prob_dict = self.net.forward(dec_seqs)
        begin_r = []
        for i in range(histories.size(0)):
            end = self._get_first_none_zero_index(dec_seqs[i])
            target = targets[i]
            ps = prob_dict[i][end]
            
            _, indices = ps.sort(dim=0, descending=True)
            indices = indices + 1 # Obtain the sorted item list
            items = self._delete_item_in_history(indices, dec_seqs[i][:end+1]) # Filter the item in the history
            ## Calculate the reverse of rank
            begin_r.append((items==target).nonzero()[0].item()+1)
            
        # Calculate new sequence rank
        dec_seqs = new_seqs[:,:-1].clone()
        prob_dict = self.net.forward(dec_seqs)
        end_r = []
        for i in range(new_seqs.size(0)):
            target = targets[i]
            end = self._get_last_path_index(dec_seqs[i], target)
            ps = prob_dict[i][end]
            
            _, indices = ps.sort(dim=0, descending=True)
            indices = indices + 1 # Obtain the sorted item list
            items = self._delete_item_in_history(indices, dec_seqs[i][:end+1]) # Filter the item in the history
                
            ## Calculate the reverse of rank
            end_r.append((items==target).nonzero()[0].item()+1)
            
        irr = np.array([1/end_r[i] - 1/begin_r[i] for i in range(len(end_r))])
        ir = np.array([end_r[i] - begin_r[i] for i in range(len(end_r))])
        return irr, ir
    
    def get_pp_in_batch(self, new_seqs, start_pos, l_paths):
        """
        Calculate the naturalness (i.e. perplexity) of the influence path

        Args:
            new_seqs (tensor): batch of item sequences containing the influence path
            start_pos (tensor): batch of starting indexes of influence paths
            l_paths (tensor): batch of the influence path lengths
            
        Returns:
            array: PPL of each sequence
        """
        pps = []
        self.net.eval()
        dec_inp_seq = new_seqs[:, :-1].clone()  # this does not remove the last eos ,  batch x seq_len-1
        output = self.net.forward(dec_inp_seq)  # [batch x (seq_len-1) x n_tokens] , vocab_size = n_tokens                                                                                          
        for i in range(new_seqs.size(0)): # Each sequence may have different starting path position
            ## Perplexity
            left_idx = start_pos[i] # The term p(first item in path|history)
            right_idx = left_idx + l_paths[i] # The term p(target|new history)
            # Obtained the target sequence
            dec_tgt_seq = new_seqs[i][left_idx:right_idx].contiguous().view(-1)

            mask = dec_tgt_seq.gt(self.PAD_ID)
            masked_tgt = dec_tgt_seq.masked_select(mask) # Masked target sequence
            # Obtain the output sequence
            output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)
            masked_output = output[i][left_idx -1 :right_idx - 1].masked_select(output_mask).view(-1, self.vocab_size)

            loss = self.loss_function(masked_output, masked_tgt-1)
            pps.append(loss.item())
        return pps


