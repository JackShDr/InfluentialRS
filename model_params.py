# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:01:02 2021

@author: ZHU Haoren
"""

# Define the sweep behaviour
#* Add parameters validation before training (e.g. n-heads divisible)
irs_sweep_config = {
    'method': 'bayes',  #bayes, grid, random
    'metric': {
        'name': 'hit',
        'goal': 'maximize',
    },
    'parameters': {
        'batch_size': {
            'values': [64, 128, 256, 512]
        },
        'emb_dim': {
            'values': [10, 20, 30, 40, 60]
        },
        'u_emb_dim': {
            'values': [2, 5, 6, 8, 10]
        },
        'lr1': {
            "min": 0.0001,
            "max": 0.01,
        },
        'n_layers': {
            'values': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        'n_heads': {
            'values': [1, 2, 3, 4, 5, 6]
        }
    },
    "early_terminate": {
        "type": "hyperband",
        "s": 2,
        "eta": 3,
        "max_iter": 20,
    }
}

tran_sweep_config = {
    'method': 'bayes', #bayes, grid, random
    'metric': {
      'name': 'val_loss_min',
      'goal': 'minimize',
    },
    'parameters': {
        'batch_size': {
            'values': [64, 128, 256, 512]
        },
        'emb_dim': {
            'values': [10, 20, 30]
        },
        'n_heads':{
            'values': [1, 2, 3, 4, 5, 6]
        },
        'lr1': {
            'values': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        },
        'n_layers':{
            'values': [1, 2, 3, 4, 5, 6, 7, 8]
        }
    },
    "early_terminate":{
       "type": "hyperband",
       "s": 2,
       "eta": 3,
       "max_iter": 20,
   }
}


# Define the model parameters for evaluator
#* Movielens-1m parameters
class NetParams():

    def __init__(self):
        # Parameters for caser
        self.params_caser = dict(
            file_class="session",
            max_len=15,  # seq_len (i.e. max_len) + 1 target
            min_len=15,
            step=1,
            d=30,
            nv=4,
            nh=16,
            drop=0.5,
            ac_conv='relu',
            ac_fc='relu',
            n_iter=50,
            batch_size=512,
            learning_rate=1e-3,
            l2=1e-6,
            neg_samples=5,
            early_stop_patience=10,
        )
        # Parameters for SASRec
        self.params_sas = dict(
            file_class="raw",
            max_len=20,
            batch_size=128,
            lr=1e-3,
            hidden_units=120,
            num_blocks=4,
            num_epochs=400,
            num_heads=3,
            dropout_rate=0.2,
            l2_emb=0.0,
            early_stop_patience=15,
        )
        # Parameters for sampleNet
        self.params_tran = dict(
            file_class="nn1",
            max_len=60,
            emb_dim=30,
            n_layers=6,
            n_heads=6,
            ffn_dim=120,
            dropout=0.2,
            lr1=0.003,
            batch_size=128,
        )
        # Parameters for pop
        self.params_pop = dict(file_class="raw")
        # Parameters for Transrec
        self.params_transrec = dict(
            file_class="raw",
            K=10,
            lr=0.05,
            max_iter=100,
            lam=0.05,
            bias_lam=0.01,
            reg_lam=0.1,
            earlystop_threshold=10,
        )
        # Parameters for BPR
        self.params_bpr = dict(
            file_class="raw",
            dim=10,
            lr=1e-3,
            weight_decay=0.025,
            n_epochs=500,
            batch_size=256,
            earlystop_threshold=1e-2,
        )
