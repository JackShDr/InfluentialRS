#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   main_evaluator.py
@Time    :   2022/01/4 13:42:19
@Author  :   Haoren Zhu 
@Contact :   hzhual@connect.ust.hk
'''
import torch
import numpy as np
import argparse

from evaluator_pipeline import eval_pipeline
from utils import str2bool, set_seed
from model_params import tran_sweep_config

### Configuration
parser = argparse.ArgumentParser()
parser.add_argument("--use_wandb", type=str2bool, default = False)
parser.add_argument("--seed", type=int, default = 17)
parser.add_argument("--project_name", type=str, default = "influential")
parser.add_argument('--dataset',type=str, default="ml-1m")
parser.add_argument("--task", type=str, default="nn1")
parser.add_argument("--method", type=str, default="td")
parser.add_argument("--datapath", type=str, default='./data/')
parser.add_argument("--model_store_path", type=str, default='./save_model/')

parser.add_argument("--first", type=str2bool, default = True)
parser.add_argument("--load", type=str2bool, default = False)
parser.add_argument("--save", type=str2bool, default = True)
parser.add_argument("--max_len", type=int, default = 60)
parser.add_argument("--min_len", type=int, default = 30)
parser.add_argument("--infrequent_thredshold", type=int, default=5)
parser.add_argument('--train_ratio',type=float, default=0.9)

parser.add_argument('--batch_size',type=int, default=128)
parser.add_argument('--epoch',type=int, default=1000)
parser.add_argument('--emb_dim',type=int, default=30)
parser.add_argument('--n_layers',type=int, default=6)
parser.add_argument('--n_heads',type=int, default=5)
parser.add_argument('--ffn_dim',type=int, default=120)
parser.add_argument('--dropout',type=float, default=0.2)
parser.add_argument('--early_stop_patience',type=int, default=15)
parser.add_argument('--lr1',type=float, default=3e-3)
parser.add_argument('--nn1_train',type=float, default=0.9)

parser.add_argument('--top_k',type=int, default=20)
parser.add_argument('--max_path_len',type=int, default=30)
args = parser.parse_args()

# Use wandb to track the model
if args.use_wandb == True:
    try:
        import wandb
        print("Use WandB to track the model...\n")
    except ImportError:
        print("WandB has not been installed...\n")
        args.use_wandb = False

# Transform the argparser config to a dictionary, which will be used by wandb later
if args.use_wandb == True:
    default_config = vars(args)

# Ensure deterministic behavior
set_seed(args.seed, cuda=torch.cuda.is_available())

# Device control
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


# Evaluator Agent
def eval_agent():
    if args.use_wandb == True:
        with wandb.init(project=args.project_name, config=default_config):
            config = wandb.config 
            eval_pipeline(config, device=device)
    else:
        eval_pipeline(args, device=device)      

if __name__ == '__main__':
    if args.task == "sweep" and args.use_wandb == True:
        sweep_config = tran_sweep_config
        sweep_id = wandb.sweep(sweep_config, project=args.project_name)
        wandb.agent(sweep_id, eval_agent())
    else: # args.task == 'normal'
        eval_agent()