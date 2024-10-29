# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:38:36 2022

@author: ZHU Haoren
"""

import sys 
sys.path.append('/home/hzhual/.conda/envs/pytorch/lib/python3.6/site-packages')
### Import package
import os
import copy
import numpy as np
import pandas as pd
import torch
import argparse
from baselines import POP, MC, FPMC, TransRec, BPR
from model.sas import SASRec
from model.caser import CaserRS
from data_provider import DataProvider
from utils import str2bool, set_seed, get_feature_vector, cal_fv_dist
from params import NetParams
from prs import evaluate_prob

### Configuration for the task
parser = argparse.ArgumentParser()
# Problem related
parser.add_argument("--use_wandb", type=str2bool, default = False)
parser.add_argument("--use_cuda", type=str2bool, default = True)
parser.add_argument("--seed", type=int, default = 1234)
parser.add_argument("--project_name", type=str, default = "influential")
parser.add_argument('--dataset',type=str, default="lastfm_small")
parser.add_argument("--task", type=str, default="method")
parser.add_argument("--method", type=str, default="pop")
parser.add_argument("--datapath", type=str, default='./data/')
parser.add_argument("--model_store_path", type=str, default='./save_model/')
parser.add_argument("--earlystop_patience", type=int, default = 10)
parser.add_argument("--is_train", type=str2bool, default = False)
parser.add_argument("--is_eval", type=str2bool, default = True)
# Dataset related
parser.add_argument("--first", type=str2bool, default = False)
parser.add_argument("--load", type=str2bool, default = False)
parser.add_argument("--save", type=str2bool, default = True)
parser.add_argument("--max_len", type=int, default = 50)
parser.add_argument("--min_len", type=int, default = 20)
parser.add_argument("--step", type=int, default = -1)
parser.add_argument("--infrequent_thredshold", type=int, default=5)
parser.add_argument('--train_ratio',type=float, default=0.9)
# Evaluation related
parser.add_argument('--top_k',type=int, default=20)
parser.add_argument('--k_c',type=int, default=1)
parser.add_argument('--max_path_len',type=int, default=20)
parser.add_argument('--use_fv',type=str2bool, default=True)
parser.add_argument('--use_h',type=str2bool, default=True)
args = parser.parse_args()

# Obtain the parameters of the model
params = NetParams()
nn1_config = getattr(params, "params_nn1")

# Use wandb to track the model
if args.use_wandb == True:
    try:
        import wandb
        print("Use WandB to track the model...\n")
    except ImportError:
        print("WandB has not been installed...\n")
        args.use_wandb = False

# Device control
device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
torch.cuda.empty_cache()
set_seed(args.seed, cuda=torch.cuda.is_available())


def cal_accuracy(preds, labels):
    """
    Calculate traditional RS metrics
    """
    hit = 0
    rr = []
    for i in range(len(labels)):
        pred = preds[i]
        label = labels[i]
        ## Calculate Hit@top_k
        if label in pred:
            hit += 1  # Increment hit count
            ## Calculate the reverse of rank
            reverse_rank = np.reciprocal(float(((pred==label).nonzero()[0].item()+1)))
            rr.append(reverse_rank)
    return hit, rr

def get_path(seqs, users, targets, net, k_c=5, max_path_len=20, fv_dict = None, binary=True):
    """
    Generate the persuasion path
    """
    # Path generation
    n_success = 0
    stop = np.zeros(len(seqs))
    paths = np.zeros((len(seqs), max_path_len))
    
    
    for i in range(max_path_len): # Iterate over maximum path length
        next_sets = net.predict_next(seqs, users, top_k = k_c)
        for j in range(len(seqs)): # Iterate over sequence sample
            if stop[j] == 1:
                continue
            seq = seqs[j]
            next_set = next_sets[j]
            target = targets[j]  
            next_dists = np.array([cal_fv_dist(k, target, fv_dict, binary) for k in next_set])
            next_item_pos = np.argsort(next_dists)[0]
            next_item = next_set[next_item_pos]

            paths[j][i] = next_item
            if next_item == target:
                n_success += 1
                stop[j] = 1
            seq = np.concatenate((seq, [next_item]))
            seqs[j] = seq
    return paths, n_success

def generate_path(config, dp):
    # Load feature vector dictionary
    fv_dict, binary = get_feature_vector(config.dataset, config.datapath, config.use_fv)
    
    
    
        
    result_save_dir = "./results/" + config.dataset + "/" + config.method + "/"
    if not os.path.isdir(result_save_dir):
        os.makedirs(result_save_dir)
    
    # Random Target Task
    eval_data  = dp.get_random_evaluate_data(seq_len = config.max_len, use_train = None, \
                                             file_class = config.file_class, gap_len = None)
#    eval_data  = dp.get_random_evaluate_data_c(seq_len = config.max_len, use_train = None, file_class = config.file_class, gap_len = None, fv_dict = fv_dict, mu = 0.2)
    
#    print(len(dp.popular_item))
    
    if config.method == "mc":
        net = MC(config)
    elif config.method == "fpmc":
        net = FPMC(config)
    elif config.method == "transrec":
        net = TransRec(config)
    elif config.method == "bpr":
        net = BPR(config)
    elif config.method == "caser":
        net = CaserRS(config, device)
    elif config.method == "sas":
        net = SASRec(config, device)
    else:
        net = POP(item_space = config.n_item)
        
    model_path = "./baselines/" + config.dataset + "/" + config.method + "/"
    net.load(model_path)
    
    # Initialize variables
    hit = 0
    rr = 0
    histories = []
    seqs = []
    users = []
    labels = []
    targets = []
    
    # Calculate accuracy related
    for i in range(len(eval_data)):
        seq = eval_data[i][0]
        history = copy.deepcopy(seq)[-config.max_len:]
        seq = torch.tensor(seq)
        user = eval_data[i][1]
        target = eval_data[i][2]
        label = eval_data[i][3]
        # Obtain the sorted list
                
        histories.append(history)
        seqs.append(seq)
        users.append(user)
        labels.append(label)
        targets.append(target)
        
#        print(min(seq))
    
    length = len(targets)
    preds = net.predict_next(seqs, users, top_k = config.top_k, use_h=config.use_h)
    hit, rr = cal_accuracy(preds, labels)

    # Obtain the paths
    paths, n_success = get_path(seqs, users, targets, net, k_c = config.k_c, max_path_len=config.max_path_len, fv_dict = fv_dict, binary=binary)
            
    
    hit = hit/length
    mrr = sum(rr)/length
    sr = n_success/length
    print("Hit@%i: %f"%(config.top_k,hit))
    print("MRR: %f"%mrr)
    print("SR: %f"%sr)
    # Save the input and the derived path
    np.save(result_save_dir+"paths_d_%i.npy"%config.max_path_len,paths)
    np.save(result_save_dir+"histories.npy",histories)
    np.save(result_save_dir+"targets_d_%i.npy"%config.max_path_len,targets)


def train_model(config, dp):
    # Generate training and testing dataset
    train, valid = dp.get_refer_data(load = config.load, save = config.save, file_class = config.file_class,\
            train_ratio = config.train_ratio, seq_len = config.max_len, min_len = config.min_len, step = config.step)
    
    if config.method == "mc":
        net = MC(config)
    elif config.method == "fpmc":
        net = FPMC(config)
    elif config.method == "transrec":
        net = TransRec(config)
    elif config.method == "bpr":
        net = BPR(config)
    elif config.method == "caser":
        net = CaserRS(config, device)
    elif config.method == "sas":
        net = SASRec(config, device)
    else:
        net = POP(item_space = config.n_item)
        
    model_save_path = "./baselines/" + args.dataset + "/" + args.method + "/"
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
        
#        
#    for i in range(len(train)):
#        if len(train[i][0])==15:
#            print(train[i][0])  
        
        
    if config.method == "caser" or config.method == "sas": # Need history for negative sampling
        candidate = dp.get_candidate_set()
        train = [train, candidate]
        
    # Train the model
    net.train(train, valid)
    # Save the model
    net.save(model_save_path)

def pipeline(config):
    dp = DataProvider(config, verbose = True)
    config.n_item = dp.n_item
    config.n_user = dp.n_user
    print("Model: %s\n"%config.method)
    # Train the baseline model
    if config.is_train == True:
        train_model(config, dp)
    
    # Obtain performance metrics and generate path
    generate_path(config, dp)
    
    print("Candidate set: %i\n"%config.k_c)
    # Use the evaluator to measure the path
    if config.is_eval == True:
        evaluate_prob(config, nn1_config, dp, device)

def baseline_agent():
    if args.use_wandb == True:
        with wandb.init(project=args.project_name, config=default_config):
            config = wandb.config 
            pipeline(config)
    else:
        pipeline(args)
        
if __name__ == "__main__":
    if args.task == "tune": # Tune one model
        model_config = getattr(params, "params_%s"%args.method)
        # Combine the task configuration and model configuration
        if args.use_wandb == True:
            default_config = {**vars(args), **model_config}
        else:
            args = argparse.Namespace(**{**vars(args), **model_config})
        baseline_agent()
    elif args.task == "method":
        methods = ["pop", "bpr", "transrec",  "caser"]
        for m in methods:
            args.method = m
            model_config = getattr(params, "params_%s"%m)
            args = argparse.Namespace(**{**vars(args), **model_config})
            baseline_agent()
    elif args.task == "candidate":
        methods = ["pop", "bpr", "transrec", "caser"]
#        methods = ["transrec", "bpr", "mc", "caser"]
        k_c = [10, 20, 30, 40, 50]
        for m in methods:
            for k in k_c:
                args.method = m
                args.k_c = k
                model_config = getattr(params, "params_%s"%m)
                args = argparse.Namespace(**{**vars(args), **model_config})
                baseline_agent()
            
