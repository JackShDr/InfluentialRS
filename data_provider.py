#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   data_provider.py
@Time    :   2023/04/29 11:55:39
@Author  :   Haoren Zhu 
@Contact :   hzhual@connect.ust.hk
'''

import torch
import pandas as pd
import numpy as np
import argparse
import json
import ast
import random

from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from collections import deque

from utils import get_end_index, get_start_index, get_item_index, str2bool


### Data Provider
class DataProvider():
    """
    Data provider for the framework
    """

    def __init__(self, config, verbose=False):
        # configuration parameteres
        self.first = config.first
        self.verbose = verbose
        self.dataset = config.dataset
        self.path = config.datapath + self.dataset + '/'
        # Generate whole data, number of users, and number of items
        self.data, self.n_user, self.n_item = self._load_data(
            self.path,
            first=config.first,
            threshold=config.infrequent_thredshold)

        self.popular_item = self._cal_popular_item_set(self.data, min_count=5)

    def _cal_popular_item_set(self, data, min_count=5):
        count = np.zeros(self.n_item)
        for seq in self.data:
            for i in seq:
                count[i - 1] += 1

        pos = np.where(count >= min_count)[0] + 1
        return set(pos)

    def _load_data(self, path, first=True, threshold=None):
        """
        Read whole data from the path. Called when creating an instance of DataProvider. 
        
        params:
            first: bool, if re-generate the sequence dataset
            threshold: int, the minimum occurence of item and user
        """
        if first == False:  # If do not re-generate the dataset
            data_pack = np.load(path + "data_pack.npy",
                                allow_pickle=True).item()
            if self.verbose == True:
                print("===== Item-User information =====")
                print("Number of User: %i" % data_pack["user_count"])
                print("Number of Item: %i" % data_pack["item_count"])
                print("============== End ==============\n")
            return data_pack["data"], data_pack["user_count"], data_pack[
                "item_count"]

        # Dataset Configuration
        if self.dataset == "ml-1m":
            rating_file = path + 'ratings.dat'
            # Load user records
            data =  pd.read_csv(rating_file, sep="::", header=None, engine="python", \
                               names=['userId', 'itemId', 'rating', 'timestamp'])
        elif self.dataset == "ml-100k":
            rating_file = path + 'u.data'
            data = pd.read_csv(
                rating_file,
                sep='\t',
                names=['userId', 'itemId', 'rating', 'timestamp'])
        elif self.dataset == "lastfm":
            rating_file = path + 'userid-timestamp-artid-artname-traid-traname.tsv'
            data = pd.read_csv(rating_file, sep = '\t', \
                names = ['userId', 'timestamp', 'artId','artName','itemId','itemName'])
        elif self.dataset == "lastfm_small":
            rating_file = path + "user_taggedartists-timestamps.dat"
            data = pd.read_csv(
                rating_file,
                sep='\t',
                names=['userId', 'itemId', 'rating', 'timestamp'])
        elif self.dataset == "epinions":
            rating_file = path + "epinions.json"
            data_str = open(rating_file).read()  # Load json string
            temp = '[%s]' % ','.join(
                data_str.splitlines())  # Replace the line split
            temp = "{'data':" + temp + "}"  # Adjust the json string
            temp = json.dumps(ast.literal_eval(
                temp))  # Replace single quote with double quote
            df = pd.read_json(temp, orient='split')
            data = df[['user', 'stars', 'time', 'item']].rename(\
                       columns={'user': 'userId', 'stars': 'rating', 'time': 'timestamp', 'item': 'itemId'})

        elif self.dataset[0:6] == "amazon":
            rating_file = path + "ratings.csv"
            data = pd.read_csv(
                rating_file,
                sep=',',
                names=['itemId', 'userId', 'rating', 'timestamp'])

        if self.dataset == "lastfm_small":
            data = data.drop_duplicates(subset=['userId', 'itemId'],
                                        keep='first')

        # Remove infrequent items and userss
        if threshold != None:
            print("Reduce to k-hop")

            stop_remove = False  # Recursively remove the infrequent item
            column_order = ["itemId", "userId"]
            while not stop_remove:
                line_count = len(data)
                for col in column_order:
                    counts = data[col].value_counts(dropna=False)
                    if sum(counts < threshold) == 0:
                        continue

                    idx = data[col].isin(counts[counts.isin(
                        list(range(threshold, counts.max())))].index)
                    data = data[idx]
                    data.reset_index(drop=True, inplace=True)

                if len(data) == line_count:
                    stop_remove = True

        # Generate user sequences, and generate mapping of item id
        sequences = []
        for userId, hist in data.groupby('userId'):
            hist = hist.sort_values('timestamp').reset_index()
            item_list = hist['itemId'].tolist()
            sequences.append(item_list)

        user_count = len(sequences)
        item_set = data['itemId'].unique()
        np.random.shuffle(item_set)
        item_count = len(item_set)

        # Generate mapping of item id
        item_id = 1
        item_dict = {}
        item_reverse_dict = {}
        for i in item_set:
            item_dict[i] = item_id
            item_reverse_dict[item_id] = i
            item_id += 1
        np.save(path + "item_dict.npy", item_dict)
        np.save(path + "item_reverse_dict", item_reverse_dict)

        # Map item in the sequences to itemId
        for i in range(user_count):
            origin_seq = sequences[i]
            sequences[i] = [item_dict[j]
                            for j in origin_seq]  # Mapping to item id

        # Print out information
        if self.verbose == True:
            # Print out item-user info
            print("===== Item-User information =====")
            if self.dataset[:2] == "ml":  # Only movielens have id type int
                print("Min User Id: %i" % min(data['userId'].unique()))
                print("Min Item Id: %i" % min(data['itemId'].unique()))
                print("Max User Id: %i" % max(data['userId'].unique()))
                print("Max Item Id: %i" % max(data['itemId'].unique()))
            print("Number of User: %i" % user_count)
            print("Number of Item: %i" % item_count)
            print("============== End ==============\n")
            # Print out sequence info
            lens = [len(i) for i in sequences]
            print("===== Sequence information ======")
            print("Number of sequences: %i" % len(sequences))
            print("Max length: %i" % max(lens))
            print("Min length: %i" % min(lens))
            print("Avg length: %i" % (sum(lens) / (len(lens))))
            print("============== End ==============\n")

        pack = {
            "user_count": user_count,
            "item_count": item_count,
            "data": sequences
        }
        np.save(path + "data_pack.npy", pack)
        return sequences, user_count, item_count

    def _nn1_split_data(self, data, seq_len=100, min_len=40):
        """
        Split the data into training and validation set. Used by transformer evaluator
        """
        # Generate subsequences from user sequences
        print("Generate dataset for transformer-based evaluator...\n")
        seqs = []
        seqs_count = 0
        seq_len += 1
        for user_id in range(len(data)):  # for every single user sequence
            # Generate training sequences
            seq = data[
                user_id][:-1]  # the remaining part is for accuracy testing
            length = len(seq)
            sub_count = int(np.ceil(length / seq_len))
            for i in range(sub_count):
                if i > 0 and i == sub_count - 1:  # more than two subsequences, and it is the last one
                    start_idx = length - min_len
                    end_idx = length
                elif i == 0 and i == sub_count - 1:  # The fisrt subsequence is the last one
                    start_idx = 0
                    end_idx = length
                else:
                    start_idx = i * seq_len
                    end_idx = (i + 1) * seq_len
                # Initialize the sequence
                new_seq = np.zeros(seq_len)
                l_seq = end_idx - start_idx
                new_seq[:l_seq] = seq[start_idx:
                                      end_idx]  # Post-padding is used
                seqs.append([
                    new_seq, user_id
                ])  # Append the sequence and the corresponding user id
                seqs_count += 1
        all_seqs = np.array(seqs)
        return all_seqs

    def _random_split_data(self, data, seq_len=100, min_len=40, step=-1):
        """
        Split the data into training and validation set. Used by IRN.  
        """
        # Generate subsequences from user sequences
        print("Generate dataset for IRN...\n")
        seqs = []
        seqs_count = 0
        for user_id in range(len(data)):  # for every single user sequence
            # Generate training sequences
            seq = data[
                user_id][:-1]  # the remaining part is for accuracy testing
            length = len(seq)  # The length of the user sequence

            stop = False  # Signal if stop generating subsequence on the user sequence
            start_idx = 0
            while not stop:
                # Locate the start and end position of the subsequence
                remain_length = length - start_idx  # Calculate the remaining length
                if remain_length <= min_len:  # If the remaining length is smaller than min_len
                    # Locate the position of start and end
                    start_idx = length - min_len
                    end_idx = length
                    stop = True
                elif remain_length < seq_len:  # If the remaining length is between min_len and max_len
                    start_idx = start_idx  # Starting position unchange
                    end_idx = np.random.randint(start_idx + min_len, length)
                else:
                    start_idx = start_idx  # Starting position unchange
                    end_idx = np.random.randint(start_idx + min_len,
                                                start_idx + seq_len)

                # Extract the subsequence and copy it to a new sequence
                sub_seq = seq[start_idx:end_idx]
                new_seq = np.zeros(seq_len)
                new_seq[-len(sub_seq):] = sub_seq  # Pre-padding
                seqs.append([
                    new_seq, user_id
                ])  # Append the sequence and the corresponding user id
                seqs_count += 1

                # Update the new starting position
                if step > 0:  # Increment by step
                    start_idx += step
                else:  # Non-overlapping segmentation
                    start_idx = end_idx

        all_seqs = np.array(seqs)
        return all_seqs

    def _session_split_data(self, data, seq_len=10, min_len=10, step=-1):
        #        print(seq_len)
        #        print(min_len)
        print("Generate session-like dataset...\n")
        seqs = []
        seqs_count = 0
        seq_len += 1
        for user_id in range(len(data)):  # for every single user sequence
            # Generate training sequences
            seq = data[
                user_id][:-1]  # the remaining part is for accuracy testing
            stop = False  # Signal if stop generating subsequence on the user sequence
            length = len(seq)  # The length of the user sequence
            start_idx = 0
            while not stop:
                # Locate the start and end position of the subsequence
                remain_length = length - start_idx  # Calculate the remaining length
                if remain_length <= seq_len:  # If the remaining length is smaller than seq len
                    end_idx = remain_length
                    stop = True
                else:
                    end_idx = start_idx + seq_len
                # Extract the subsequence and copy it to a new sequence
                new_seq = seq[start_idx:end_idx]
                #                print(end_idx-start_idx)
                if len(new_seq) < min_len + 1:
                    continue

#                print(len(new_seq))
                seqs.append([
                    new_seq, user_id
                ])  # Append the sequence and the corresponding user id
                seqs_count += 1
                # Update the new starting position
                if step > 0:  # Increment by step
                    start_idx += step
                else:  # Non-overlapping segmentation
                    start_idx = end_idx
        all_seqs = np.array(seqs)
        return all_seqs

    def _split_data(self, data):
        print("Generate raw sequence dataset...\n")
        seqs = [d[:-1] for d in data]
        all_seqs = np.array(seqs)
        return all_seqs, None

    def get_candidate_set(self):
        candidate_set = []
        for i in range(len(self.data)):
            h = set(self.data[i][:-1])
            item_space = set(np.arange(self.n_item) + 1)
            negative_space = list(item_space - h)
            candidate_set.append(negative_space)
        return candidate_set

    def get_refer_data(self, load = False, save = False, file_class = "irs", train_ratio = 0.9, \
                       seq_len = 100, min_len = 40,  step = -1):
        """
        Generate training data and validation data 
        
        params:
            file_class: string, the format of training and testing data
        """
        # Obtain data
        if load == False:  # If do not load existing dataset
            # Generate all the sequence data
            if file_class == "irs":
                all_seqs = self._random_split_data(data=self.data,
                                                   seq_len=seq_len,
                                                   min_len=min_len,
                                                   step=step)
            elif file_class == "session":
                all_seqs = self._session_split_data(data=self.data,
                                                    seq_len=seq_len,
                                                    min_len=min_len,
                                                    step=step)
            elif file_class == "nn1":
                all_seqs = self._nn1_split_data(data=self.data,
                                                seq_len=seq_len,
                                                min_len=min_len)
            else:  # file_class is "raw"
                return self._split_data(data=self.data)

            # Split to training, validation
            np.random.shuffle(all_seqs)
            train_split_idx = int(len(all_seqs) * train_ratio)
            valid_seqs = all_seqs[train_split_idx:]
            train_seqs = all_seqs[:train_split_idx]
            if save == True:  # Saving data
                np.save(self.path + "%s_train_seq.npy" % file_class,
                        train_seqs)
                np.save(self.path + "%s_valid_seq.npy" % file_class,
                        valid_seqs)
                np.save(self.path + "%s_all_seq.npy" % file_class, all_seqs)

            if file_class == "irs":  # Save the targets that used in the training dataset
                targets = list(set([seq[0][-1] for seq in train_seqs]))
                np.save(self.path + "%s_targets.npy" % file_class, targets)
        else:  # If load existing dataset
            print("Load dataset...\n")
            train_seqs = np.load(self.path + "%s_train_seq.npy" % file_class,
                                 allow_pickle=True)
            valid_seqs = np.load(self.path + "%s_valid_seq.npy" % file_class,
                                 allow_pickle=True)

        # Print out split dataset information
        if self.verbose == True:
            print("=== Split dataset information ===")
            print("Train length: %i" % len(train_seqs))
            print("Val length: %i" % len(valid_seqs))
            print("============== End ==============\n")

        return train_seqs, valid_seqs

    def get_random_evaluate_data(self,
                                 seq_len=100,
                                 use_train=False,
                                 file_class="irs",
                                 gap_len=20,
                                 use_popular=True):
        """
        Process the sequence data for testing random target
        """
        if use_train == True and file_class == "irs":
            item_set = set(np.load(self.path + "irs_targets.npy"))
        else:
            item_set = set([i + 1 for i in range(self.n_item)])

        if use_popular == True:
            item_set = item_set.intersection(self.popular_item)


#        targets = []
#        labels = [] # Groundtruth labels for accuracy results
#        new_seqs = []
#        users = []
        eval_data = []

        for user_id in range(len(self.data)):  # For each user
            # Generate the viewing history
            seq = self.data[user_id][:-1]
            label = self.data[user_id][-1]
            # Sample a target item that does not exist in the viewing history
            history = set(seq[-seq_len:])  # Read only the recent history
            space = item_set.difference(history)
            random_target = random.sample(space, 1)[0]

            #            # Generate testing sequence format
            #            if file_class == "irs":
            #                new_seq = np.zeros(seq_len)
            #                # History length is (seq_len - path_len - 1), plus one target item and path space
            #                l_history = seq_len - gap_len -1
            #                item_list = seq[-l_history:]
            #                start_history = -len(item_list) - gap_len -1 # History start index
            #                new_seq[start_history:start_history + len(item_list)] = item_list # Pre-padding
            #                new_seq[-1] = random_target # Final new_seq: [0, 0, histoy, 0, 0, target]
            #            else:
            new_seq = seq[-seq_len:]  # The whole sequence as history

            #            targets.append(random_target)
            #            labels.append(label)
            #            new_seqs.append(new_seq)
            #            users.append(user_id)

            eval_data.append([new_seq, user_id, random_target, label])
        return eval_data

    def get_random_evaluate_data_c(self,
                                   seq_len=100,
                                   use_train=False,
                                   file_class="irs",
                                   gap_len=20,
                                   use_popular=True,
                                   fv_dict=None,
                                   mu=None):
        """
        Process the sequence data for testing random target with constraints
        """
        # Process the target space
        if use_train == True and file_class == "irs":
            item_set = set(np.load(self.path + "irs_targets.npy"))
        else:
            item_set = set([i + 1 for i in range(self.n_item)])

        if use_popular == True:
            item_set = item_set.intersection(self.popular_item)

        # Obtain the testing data
        eval_data = []

        for user_id in range(len(self.data)):  # For each user
            # Generate the viewing history
            seq = self.data[user_id][:-1]
            label = self.data[user_id][-1]
            # Sample a target item that does not exist in the viewing history
            history = set(seq[-seq_len:])  # Read only the recent history
            space = item_set.difference(history)

            if fv_dict is not None and mu is not None:
                # Obtain user embedding
                embs = []
                for i in history:
                    if i != 0:
                        embs.append(fv_dict[int(i)])
                embs = np.array(embs)
                u_emb = np.mean(embs, axis=0)

                # Obtain target embedding
                find_t = True
                while find_t:
                    random_target = random.sample(space, 1)[0]
                    t_emb = fv_dict[int(random_target)]
                    print(np.linalg.norm(u_emb - t_emb))
                    if np.linalg.norm(u_emb - t_emb) < mu:
                        find_t = False

            new_seq = seq[-seq_len:]  # The whole sequence as history
            eval_data.append([new_seq, user_id, random_target, label])
        return eval_data


class DatasetNN(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class DataLoaderNN1(DataLoader):
    """
    Class for NN1 dataloader  
    """

    def __init__(self, *args, **kwargs):
        super(DataLoaderNN1, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        seqs = []
        users = []
        for row in batch:
            seqs.append(row[0])
            users.append(row[1])
        return torch.LongTensor(np.array(seqs)), torch.LongTensor(users)


class DataLoaderIRS(DataLoader):
    """
    Class for IRN dataloader  
    """

    def __init__(self, *args, **kwargs):
        super(DataLoaderIRS, self).__init__(*args, **kwargs)
        self.random_target = False
        self.collate_fn = self._collate_fn if self.random_target == False else self._random_collate_fn

    def _random_collate_fn(self, batch):
        seqs = []
        users = []
        for row in batch:
            seq = np.array(row[0])
            start_pos = get_start_index(
                seq
            )  # Find the first non-zero, since the sequence is prepadding
            length = len(
                seq) - start_pos  # Find the length of the actual sequence
            # The window of sampling target item
            left_idx = int(length * 0.67) + start_pos
            right_idx = len(seq)
            target_pos = np.random.randint(left_idx, right_idx)
            # Obtain the new sequence
            item_list = seq[start_pos:target_pos + 1]
            new_seq = np.zeros(len(seq))
            new_seq[-len(item_list):] = item_list
            # Add to the batch
            seqs.append(new_seq)
            users.append(row[1])
        return torch.LongTensor(np.array(seqs)), torch.LongTensor(users)

    def _collate_fn(self, batch):
        seqs = []
        users = []
        for row in batch:
            seqs.append(row[0])
            users.append(row[1])

        return torch.LongTensor(np.array(seqs)), torch.LongTensor(users)


class DataLoaderEvalIRS(DataLoader):
    """
    Class for evaluating IRN dataloader 
    
    Shape of eval_data: [new_seq, user_id, random_target, label]
    """

    def __init__(self, seq_len, gap_len, *args, **kwargs):
        super(DataLoaderEvalIRS, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.seq_len = seq_len
        self.gap_len = gap_len

    def _collate_fn(self, batch):
        raw_seqs = []
        seqs = []
        users = []
        targets = []
        labels = []
        for row in batch:
            seq = row[0]
            target = row[2]

            new_seq = np.zeros(self.seq_len)
            # History length is (seq_len - path_len - 1), plus one target item and path space
            l_history = self.seq_len - self.gap_len - 1
            item_list = seq[-l_history:]
            start_history = -len(
                item_list) - self.gap_len - 1  # History start index
            new_seq[start_history:start_history +
                    len(item_list)] = item_list  # Pre-padding
            new_seq[-1] = target  # Final new_seq: [0, 0, histoy, 0, 0, target]

            raw_seqs.append(torch.LongTensor(seq))
            seqs.append(new_seq)
            users.append(row[1])
            targets.append(row[2])
            labels.append(row[3])
        return raw_seqs, torch.LongTensor(np.array(seqs)), torch.LongTensor(
            users), torch.LongTensor(targets), torch.tensor(labels),


class DataLoaderSession(DataLoader):

    def __init__(self, data, batch_size=64, get_path=False):
        """
        A class for creating session-parallel mini-batches.
        Args:
             dataset (SessionDataset): the session dataset to generate the batches from
             batch_size (int): size of the batch
        """
        self.data = np.array(data)
        self.batch_size = batch_size
        self.get_path = get_path

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.
        Yields:
            input (B,): torch.FloatTensor. Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """
        # initializations

        iters = np.arange(self.batch_size)
        maxiter = iters.max()

        seqs = self.data[iters]
        lengths = np.array([len(d) for d in seqs])
        starts = np.zeros(self.batch_size).astype(np.int)

        mask = []  # indicator for the sessions to be terminated
        finished = False
        n_data = len(self.data)
        if not self.get_path:
            while not finished:
                minlen = (lengths - starts).min()
                # Item indices(for embedding) for clicks where the first sessions start
                idx_target = np.array(
                    [seqs[j][starts[j]] for j in range(self.batch_size)])
                for i in range(minlen - 1):
                    # Build inputs & targets
                    idx_input = idx_target
                    idx_target = np.array([
                        seqs[j][starts[j] + i + 1]
                        for j in range(self.batch_size)
                    ])
                    input = torch.LongTensor(idx_input)
                    target = torch.LongTensor(idx_target)
                    yield input, target, mask

                # see if how many sessions should terminate
                starts = starts + (minlen - 1)
                mask = np.arange(self.batch_size)[(lengths - starts) <= 1]
                for idx in mask:
                    maxiter += 1
                    if maxiter == n_data:
                        finished = True
                        break
                    # update the next starting/ending point
                    iters[idx] = maxiter
                    starts[idx] = 0
                    seqs[idx] = self.data[maxiter]
                    lengths[idx] = len(seqs[idx])
        else:
            while not finished:
                minlen = (lengths - starts).min()
                # Item indices(for embedding) for clicks where the first sessions start
                for i in range(minlen):
                    # Build inputs & targets
                    idx_input = np.array([
                        seqs[j][starts[j] + i] for j in range(self.batch_size)
                    ])
                    input = torch.LongTensor(idx_input)
                    ends = ((starts + i + 1) == lengths)
                    yield input, mask, iters, ends

                # see if how many sessions should terminate
                starts = starts + minlen
                mask = np.arange(self.batch_size)[(lengths - starts) <= 0]
                for idx in mask:
                    maxiter += 1
                    if maxiter == n_data:
                        finished = True
                        break
                    # update the next starting/ending point
                    iters[idx] = maxiter
                    starts[idx] = 0
                    seqs[idx] = self.data[maxiter]
                    lengths[idx] = len(seqs[idx])


### Classes and functions for evaluation using evaluator
class DatasetEvalNN1(Dataset):

    def __init__(self, histories, paths, targets, seq_len=100):
        self.data = self._preprocess_seqs(histories, paths, targets, seq_len)

    def _preprocess_seqs(self, histories, paths, targets, seq_len):
        data = []
        for i in range(len(targets)):
            # Obtain the sequence
            path = paths[i]
            history = histories[i]
            target = targets[i]

            # Find the actual path
            path = path[path > 0]
            # Check if the target has arrived earlier
            pos = get_item_index(path, target)
            if pos == -1:  # Not early success
                path = np.concatenate(
                    (path, [target]))  # Append target directly to the tail

            # Obtain the new sequence with history and path
            l_path = len(path)  # The length of the path
            l_seq = seq_len + 1  # The format length of evaluator input
            new_seq = np.zeros(l_seq)
            actual_seq = np.concatenate((np.array(history), path))
            l_actual = len(actual_seq)  # The length of actual sequence
            if l_actual <= l_seq:  # If the actual length is no larger than the input length
                new_seq[:l_actual] = actual_seq
                start_pos = len(history)  # The starting position of the path
            else:  # larger
                new_seq[:] = actual_seq[-l_seq:]
                start_pos = l_seq - len(path)

            pad_history = np.zeros(l_seq)
            if len(
                    history
            ) <= seq_len:  # If the history length is no larger than the input length
                pad_history[:len(history)] = history
            else:  # larger
                pad_history[:-1] = history[-seq_len:]

            data.append([pad_history, new_seq, target, start_pos, l_path])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class DataLoaderEvalNN1(DataLoader):

    def __init__(self, *args, **kwargs):
        super(DataLoaderEvalNN1, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        histories = []
        new_seqs = []
        targets = []
        start_pos = []
        l_path = []
        for row in batch:
            histories.append(row[0])
            new_seqs.append(row[1])
            targets.append(row[2])
            start_pos.append(row[3])
            l_path.append(row[4])
        return torch.LongTensor(np.array(histories)), torch.LongTensor(
            np.array(new_seqs)), torch.tensor(targets), torch.tensor(
                start_pos), torch.tensor(l_path)


class TripletUniformPair(IterableDataset):

    def __init__(self, num_item, user_list, pair, shuffle, num_epochs):
        self.num_item = num_item
        self.user_list = user_list
        self.pair = pair
        self.shuffle = shuffle
        self.num_epochs = num_epochs

    def __iter__(self):
        worker_info = None
        # Shuffle per epoch
        self.example_size = self.num_epochs * len(self.pair)
        self.example_index_queue = deque([])
        self.seed = 0
        if worker_info is not None:
            self.start_list_index = worker_info.id
            self.num_workers = worker_info.num_workers
            self.index = worker_info.id
        else:
            self.start_list_index = None
            self.num_workers = 1
            self.index = 0
        return self

    def __next__(self):
        if self.index >= self.example_size:
            raise StopIteration
        # If `example_index_queue` is used up, replenish this list.
        while len(self.example_index_queue) == 0:
            index_list = list(range(len(self.pair)))
            if self.shuffle:
                random.Random(self.seed).shuffle(index_list)
                self.seed += 1
            if self.start_list_index is not None:
                index_list = index_list[self.start_list_index::self.
                                        num_workers]
                # Calculate next start index
                self.start_list_index = (
                    self.start_list_index +
                    (self.num_workers -
                     (len(self.pair) % self.num_workers))) % self.num_workers
            self.example_index_queue.extend(index_list)
        result = self._example(self.example_index_queue.popleft())
        self.index += self.num_workers
        return result

    def _example(self, idx):
        u = self.pair[idx][0]
        i = self.pair[idx][1]
        j = np.random.randint(self.num_item)
        while j in self.user_list[u]:
            j = np.random.randint(self.num_item)
        return u, i, j


### Testing DataProvider
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training dataset generation
    parser.add_argument("--first", type=str2bool, default=True)
    parser.add_argument('--dataset', type=str, default="ml-1m")
    parser.add_argument('--datapath', type=str, default='./data/')
    parser.add_argument("--load", type=str2bool, default=False)
    parser.add_argument("--save", type=str2bool, default=False)
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--min_len", type=int, default=30)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--infrequent_thredshold", type=int, default=5)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    # Testing dataset generation
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--gap_len', type=int, default=20)
    parser.add_argument('--use_train', type=str2bool, default=False)
    config = parser.parse_args()
    dp = DataProvider(config, verbose=True)
#    train_data, val_data  = dp.get_refer_data(load = config.load, save = config.save, file_class = "session",\
#                      train_ratio = config.train_ratio, seq_len = config.max_len, min_len = config.min_len, step = config.step)
#
#
#    train_data = DatasetNN(train_data)
#    train_dataloader = DataLoaderSession(train_data, batch_size=64)
#    eval_data  = dp.get_random_evaluate_data(seq_len = config.max_len, use_train = True, file_class = "irs", gap_len = 0)
