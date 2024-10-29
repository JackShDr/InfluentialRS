# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:59:09 2022
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import str2bool, EarlyStopping, delete_item_in_history, DatasetNN
import numpy as np

activation_getter = {
    'iden': lambda x: x,
    'relu': F.relu,
    'tanh': torch.tanh,
    'sigm': torch.sigmoid
}


class DataLoaderCaser(DataLoader):
    """
    Class for PrsNN dataloader  
    """

    def __init__(self, candidate, neg_sample, *args, **kwargs):
        super(DataLoaderCaser, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.candidate = candidate
        self.neg_sample = neg_sample

    def _collate_fn(self, batch):
        users = []
        seqs = []
        rats = []
        targets = []
        negative_samples = []

        for row in batch:
            users.append(row[0])
            seqs.append(row[1][:-1])
            rats.append(row[2][:-1] > 2)
            targets.append(row[1][-1])

            space = self.candidate[row[0]]

            negative_sample = np.random.choice(space, self.neg_sample)
            negative_samples.append(negative_sample)

        users = torch.LongTensor(users)
        seqs = torch.LongTensor(seqs)
        rats = torch.LongTensor(rats)
        targets = torch.unsqueeze(torch.LongTensor(targets), dim=1)
        negs = torch.LongTensor(negative_samples)

        return users, seqs, rats, targets, negs


class Caser(nn.Module):
    """
    Convolutional Sequence Embedding Recommendation Model (Caser)[1].
    [1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18
    Parameters
    ----------
    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, num_users, num_items, model_args):
        super(Caser, self).__init__()
        self.args = model_args

        # init args
        L = self.args.max_len  # Input length
        dims = self.args.d
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items + 1, dims, padding_idx=0)
        self.rating_embeddings = nn.Embedding(2, dims)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList(
            [nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items + 1, dims + dims, padding_idx=0)
        self.b2 = nn.Embedding(num_items + 1, 1, padding_idx=0)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(
            0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(
            0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.cache_x = None

    def forward(self, seq_var, rat_var, user_var, item_var, for_pred=False):
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets).
        Parameters
        ----------
        seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_var: torch.LongTensor with size [batch_size]
            a batch of items
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """

        # Embedding Look-up
        # use unsqueeze() to get 4-D
        item_embs = self.item_embeddings(seq_var).unsqueeze(1)
        rat_embs = self.rating_embeddings(rat_var).unsqueeze(1)
        item_embs = item_embs + rat_embs

        user_emb = self.user_embeddings(user_var).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)
        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            res = (x * w2).sum(1) + b2
        else:
            res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()

        return res


class CaserRS(object):
    """
    Contains attributes and methods that needed to train a sequential
    recommendation model. Models are trained by many tuples of
    (users, sequences, targets, negatives) and negatives are from negative
    sampling: for any known tuple of (user, sequence, targets), one or more
    items are randomly sampled to act as negatives.
    Parameters
    ----------
    n_iter: int,
        Number of iterations to run.
    batch_size: int,
        Minibatch size.
    l2: float,
        L2 loss penalty, also known as the 'lambda' of l2 regularization.
    neg_samples: int,
        Number of negative samples to generate for each targets.
        If targets=3 and neg_samples=3, then it will sample 9 negatives.
    learning_rate: float,
        Initial learning rate.
    use_cuda: boolean,
        Run the model on a GPU or CPU.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, config=None, device=None):
        # model related
        self._num_items = config.n_item
        self._num_users = config.n_user
        self._net = None
        self.model_args = config
        self.path = "./baselines/" + config.dataset + "/" + config.method + "/"

        # learning related
        self._batch_size = config.batch_size
        self._n_iter = config.n_iter
        self._learning_rate = config.learning_rate
        self._l2 = config.l2
        self._neg_samples = config.neg_samples
        self._device = device
        self.early_stop_patience = config.early_stop_patience

        # rank evaluation related
        self._candidate = []

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self):
        self._net = Caser(self._num_users, self._num_items,
                          self.model_args).to(self._device)

        self._optimizer = optim.Adam(self._net.parameters(),
                                     weight_decay=self._l2,
                                     lr=self._learning_rate)

    def _padding_sequence(self, data):
        for i in range(len(data)):
            seq = data[i][1]
            if len(seq) < self.model_args.max_len + 1:
                new_seq = np.zeros(self.model_args.max_len + 1)
                new_seq[:len(seq)] = seq
                data[i][1] = new_seq
        return data

    def save(self, path):
        pass

    def load(self, path):
        model_info = torch.load(path + '/caser_params.pth.tar')
        if not self._initialized:  # Save the valid data
            self._initialize()
        self._net.load_state_dict(model_info['state_dict'])
        self._optimizer.load_state_dict(model_info['optimizer'])

    def predict_next(self, seqs, rats, users=None, top_k=50, use_h=None):
        #        test_data = torch.zeros(len(seqs), self.model_args.max_len)
        #        for i in range(len(seqs)):
        #            test_data[i][:] = seqs[i][-self.model_args.max_len:]
        test_data = [seq[-self.model_args.max_len:] for seq in seqs]
        test_rat = [rat[-self.model_args.max_len:] > 2 for rat in rats]

        preds = torch.zeros((len(seqs), top_k))
        self._net.eval()
        with torch.no_grad():
            for i in range(len(seqs)):
                pad_seq = torch.zeros(self.model_args.max_len)
                pad_seq[-len(test_data[i]):] = torch.LongTensor(test_data[i])

                seq = torch.unsqueeze(pad_seq, dim=0).long()
                rat = torch.unsqueeze(torch.LongTensor(test_rat[i]),
                                      dim=0).long()

                item_ids = (torch.arange(self._num_items) + 1).long()
                user_id = torch.from_numpy(np.array([[users[i]]])).long()

                user, sequences, ratings, items = (user_id.to(self._device),
                                                   seq.to(self._device),
                                                   rat.to(self._device),
                                                   item_ids.to(self._device))

                out = self._net(sequences, ratings, user, items,
                                for_pred=True).detach().cpu()

                _, indices = out.sort(
                    descending=True)  # Obtain the top-k item list
                indices += 1
                if use_h == True:
                    indices = delete_item_in_history(indices,
                                                     torch.tensor(seqs[i]))

                preds[i][:] = indices[:top_k]
        return preds

    def train(self, train_data, valid_data, candidate):

        train_data = self._padding_sequence(train_data)
        valid_data = self._padding_sequence(valid_data)

        if not self._initialized:  # Save the valid data
            self._initialize()

        start_epoch = 0
        train_dataset = DatasetNN(train_data)
        valid_dataset = DatasetNN(valid_data)
        train_data_loader = DataLoaderCaser(candidate=candidate,
                                            neg_sample=self._neg_samples,
                                            dataset=train_dataset,
                                            batch_size=self._batch_size,
                                            shuffle=True,
                                            num_workers=0)
        valid_data_loader = DataLoaderCaser(candidate=candidate,
                                            neg_sample=self._neg_samples,
                                            dataset=valid_dataset,
                                            batch_size=self._batch_size,
                                            shuffle=False,
                                            num_workers=0)

        early_stopping = EarlyStopping(patience=self.early_stop_patience,
                                       verbose=True)

        for epoch_num in range(start_epoch, self._n_iter):
            # set model to training mode
            self._net.train()
            epoch_loss = 0.0

            for (i, (users, seqs, rats, targets,
                     negs)) in enumerate(train_data_loader):
                seqs = seqs.to(self._device)
                users = users.to(self._device)
                rats = rats.to(self._device)
                targets = targets.to(self._device)
                negatives = negs.to(self._device)

                items_to_predict = torch.cat((targets, negatives), 1)
                items_prediction = self._net(seqs, rats, users,
                                             items_to_predict)

                (targets_prediction, negatives_prediction) = torch.split(items_prediction,\
                                                     [targets.size(1), negatives.size(1)], dim=1)

                self._optimizer.zero_grad()
                # compute the binary cross-entropy loss
                positive_loss = -torch.mean(
                    torch.log(torch.sigmoid(targets_prediction)))
                negative_loss = -torch.mean(
                    torch.log(1 - torch.sigmoid(negatives_prediction)))
                loss = positive_loss + negative_loss
                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            epoch_loss /= i + 1

            with torch.no_grad():
                valid_loss = 0.0
                for (i, (users, seqs, rats, targets,
                         negs)) in enumerate(valid_data_loader):
                    seqs = seqs.to(self._device)
                    users = users.to(self._device)
                    rats = rats.to(self._device)
                    targets = targets.to(self._device)
                    negatives = negs.to(self._device)
                    items_to_predict = torch.cat((targets, negatives), 1)
                    items_prediction = self._net(seqs, rats, users,
                                                 items_to_predict)

                    (targets_prediction, negatives_prediction) = torch.split(items_prediction,\
                                                         [targets.size(1), negatives.size(1)], dim=1)

                    self._optimizer.zero_grad()
                    # compute the binary cross-entropy loss
                    positive_loss = -torch.mean(
                        torch.log(torch.sigmoid(targets_prediction)))
                    negative_loss = -torch.mean(
                        torch.log(1 - torch.sigmoid(negatives_prediction)))
                    loss = positive_loss + negative_loss
                    valid_loss += loss.item()

                valid_loss /= i + 1
            print("Epoch %i Loss: %f; Valid loss: %f " %
                  (epoch_num, epoch_loss, valid_loss))

            model_dict = {
                'state_dict': self._net.state_dict(),
                'optimizer': self._optimizer.state_dict()
            }
            if epoch_num > 10:  # Todo: Hyperparameter: min_epoch
                early_stopping(valid_loss, model_dict, epoch_num,
                               self.path + '/caser_params.pth.tar')
                if early_stopping.early_stop:
                    print("Early stopping...")
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_root',
                        type=str,
                        default='datasets/ml1m/test/train.txt')
    parser.add_argument('--test_root',
                        type=str,
                        default='datasets/ml1m/test/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=True)

    config = parser.parse_args()

    # model dependent arguments
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=50)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')

    model_config = model_parser.parse_args()
    model_config.L = config.L

    print(config)
    print(model_config)
