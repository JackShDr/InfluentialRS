# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 22:45:27 2022
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from utils import EarlyStopping, delete_item_in_history, DatasetNN


class DataLoaderSAS(DataLoader):
    """
    Class for PrsNN dataloader  
    """

    def __init__(self, candidate, max_len, *args, **kwargs):
        super(DataLoaderSAS, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.candidate = candidate
        self.max_len = max_len
        self.for_pred = True if candidate == None else False

    def _collate_fn(self, batch):
        # For prediction
        if self.for_pred == True:
            seqs = torch.zeros(len(batch), self.max_len).long()
            ratings = torch.zeros(len(batch), self.max_len).long()
            users = []
            for i in range(len(batch)):
                row = batch[i]
                
                user = row[0]
                seq = torch.LongTensor(row[1][-self.max_len:])
                rat = torch.LongTensor(row[2][-self.max_len:]>2)
                seq_len = len(seq)
                
                seqs[i][-seq_len:] = seq
                ratings[i][-seq_len:] = rat
                users.append(user)
            return torch.LongTensor(seqs), torch.LongTensor(
                ratings), torch.LongTensor(users)

        # For training
        seqs = torch.zeros(len(batch), self.max_len).long()
        users = []
        positive_samples = torch.zeros(len(batch), self.max_len).long()
        negative_samples = torch.zeros(len(batch), self.max_len).long()
        ratings = torch.zeros(len(batch), self.max_len).long()
        for i in range(len(batch)):
            row = batch[i]
            # Obtain sequences and postive samples

            user = row[0]
            seq = torch.LongTensor(row[1][:-1])
            rat = torch.LongTensor(np.array(row[2][:-1]) > 2)

            seq_len = len(seq)
            pos = torch.LongTensor(row[1][-seq_len:])

            space = self.candidate[user]
            negative_sample = np.random.choice(space, seq_len)

            users.append(user)
            seqs[i][-seq_len:] = seq
            positive_samples[i][-seq_len:] = pos
            ratings[i][-seq_len:] = rat
            negative_samples[i][-seq_len:] = torch.LongTensor(negative_sample)

        return torch.LongTensor(seqs), torch.LongTensor(
            ratings), torch.LongTensor(users), torch.LongTensor(
                positive_samples), torch.LongTensor(negative_samples)


class PointWiseFeedForward(torch.nn.Module):

    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(
                self.relu(self.dropout1(self.conv1(inputs.transpose(-1,
                                                                    -2))))))
        outputs = outputs.transpose(-1,
                                    -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py


class SAS(torch.nn.Module):

    def __init__(self, config=None, device=None):
        super(SAS, self).__init__()

        self.user_num = config.n_user
        self.item_num = config.n_item
        self.dev = device
        self.args = config

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1,
                                           config.hidden_units,
                                           padding_idx=0)
        self.rat_emb = torch.nn.Embedding(2, config.hidden_units)
        self.pos_emb = torch.nn.Embedding(config.max_len,
                                          config.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=config.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList(
        )  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(config.hidden_units, eps=1e-8)

        for _ in range(config.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(config.hidden_units,
                                                    eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                config.hidden_units, config.num_heads, config.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(config.hidden_units,
                                                   eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(config.hidden_units,
                                                 config.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs, rat_seqs):
        #        print(log_seqs.shape)
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim**0.5
        #        print(seqs.shape)
        positions = np.tile(np.array(range(log_seqs.shape[1])),
                            [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs += self.item_emb(torch.LongTensor(rat_seqs).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q,
                                                      seqs,
                                                      seqs,
                                                      attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, rat_seqs, pos_seqs,
                neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs,
                                   rat_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self,
                user_ids,
                log_seqs,
                rat_seqs,
                item_indices=[]):  # for inference
        log_feats = self.log2feats(log_seqs,
                                   rat_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:,
                               -1, :]  # only use last QKV classifier, a waste

        if len(item_indices) == 0:
            item_indices = torch.arange(self.item_num) + 1
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(
            self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)


class SASRS(object):

    def __init__(self, config=None, device=None):
        # model related
        self.n_item = config.n_item
        self.n_user = config.n_user
        self.max_len = config.max_len
        self.lr = config.lr
        self.l2_emb = config.l2_emb
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.net = None
        self.args = config
        self.path = "./baselines/" + config.dataset + "/" + config.method + "/"
        self.device = device
        self.early_stop_patience = config.early_stop_patience

    @property
    def _initialized(self):
        return self.net is not None

    def _initialize(self):
        self.net = SAS(self.args, self.device).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr,
                                    betas=(0.9, 0.98))
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,factor=0.5,\
                                                          patience=4,verbose=True)

    def save(self, path):
        pass

    def load(self, path):
        model_info = torch.load(path + '/sas_params.pth.tar')
        if not self._initialized:  # Save the valid data
            self._initialize()
        self.net.load_state_dict(model_info['state_dict'])
        self.optimizer.load_state_dict(model_info['optimizer'])

    def train(self, train_data, valid_data, candidate):
        if not self._initialized:  # Save the valid data
            self._initialize()

        train_dataset = DatasetNN(train_data)
        valid_dataset = DatasetNN(valid_data)
        train_data_loader = DataLoaderSAS(candidate=candidate,
                                          dataset=train_dataset,
                                          max_len=self.max_len,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=0)
        valid_data_loader = DataLoaderSAS(candidate=candidate,
                                          dataset=valid_dataset,
                                          max_len=self.max_len,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=0)

        start_epoch = 0
        early_stopping = EarlyStopping(patience=self.early_stop_patience,
                                       verbose=True)

        for epoch in range(start_epoch, self.num_epochs + 1):
            self.net.train()
            epoch_loss = 0.0
            for i, (seq, rat, u, pos, neg) in enumerate(train_data_loader):
                u, seq, rat, pos, neg = np.array(u), np.array(seq), np.array(
                    rat), np.array(pos), np.array(neg)
                pos_logits, neg_logits = self.net(u, seq, rat, pos, neg)
                pos_labels, neg_labels = torch.ones(
                    pos_logits.shape,
                    device=self.device), torch.zeros(neg_logits.shape,
                                                     device=self.device)
                # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
                self.optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = self.criterion(pos_logits[indices], pos_labels[indices])
                loss += self.criterion(neg_logits[indices],
                                       neg_labels[indices])
                for param in self.net.item_emb.parameters():
                    loss += self.l2_emb * torch.norm(param)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= i + 1

            print("Epoch %i Loss: %f" % (epoch, epoch_loss))

            with torch.no_grad():
                valid_loss = 0.0
                for i, (seq, rat, u, pos, neg) in enumerate(valid_data_loader):
                    u, seq, rat, pos, neg = np.array(u), np.array(
                        seq), np.array(rat), np.array(pos), np.array(neg)
                    pos_logits, neg_logits = self.net(u, seq, rat, pos, neg)
                    pos_labels, neg_labels = torch.ones(
                        pos_logits.shape,
                        device=self.device), torch.zeros(neg_logits.shape,
                                                         device=self.device)
                    # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
                    self.optimizer.zero_grad()
                    indices = np.where(pos != 0)
                    loss = self.criterion(pos_logits[indices],
                                          pos_labels[indices])
                    loss += self.criterion(neg_logits[indices],
                                           neg_labels[indices])
                    for param in self.net.item_emb.parameters():
                        loss += self.l2_emb * torch.norm(param)

                    valid_loss += loss.item()

                valid_loss /= i + 1

            model_dict = {
                'state_dict': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }

            if epoch > 10:  # Todo: Hyperparameter: min_epoch

                early_stopping(valid_loss, model_dict, epoch,
                               self.path + '/sas_params.pth.tar')
                if early_stopping.early_stop:
                    print("Early stopping...")
                    break

    def predict_next(self, seqs, rats, users=None, top_k=50, use_h=None):
        preds = torch.zeros((len(seqs), top_k))

        test_data = [[i, seqs[i], rats[i]] for i in range(len(seqs))]
        test_dataset = DatasetNN(test_data)
        test_data_loader = DataLoaderSAS(candidate = None, dataset = test_dataset, max_len = self.max_len,\
                                          batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.net.eval()
        outs = []
        with torch.no_grad():
            for i, (seq, rat, u) in enumerate(test_data_loader):
                u, seq, rat = np.array(u), np.array(seq), np.array(rat)
                predictions = self.net.predict(u, seq,
                                               rat).detach().cpu().numpy()
                if len(outs) == 0:
                    outs = predictions
                else:
                    outs = np.vstack((outs, predictions))

        outs = torch.LongTensor(outs)
        for i in range(len(seqs)):
            out = outs[i]
            _, indices = out.sort(
                descending=True)  # Obtain the top-k item list
            indices += 1
            if use_h == True:
                indices = delete_item_in_history(indices,
                                                 torch.tensor(seqs[i]))
            preds[i][:] = indices[:top_k]

        return preds
