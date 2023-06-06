#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   evaluator_pipeline.py
@Time    :   2021/03/29 16:57:00
@Author  :   Haoren Zhu 
@Contact :   hzhual@connect.ust.hk
'''

import torch
import torch.nn as nn
import os
import numpy as np
import wandb

from data_provider import DataProvider, DatasetNN, DataLoaderNN1
from model.evaluator import Evaluator
from model.uRS import SampleNet
from utils import EarlyStopping


def train_evaluator(config, dp, device):
    train_data, val_data = dp.get_refer_data(load=config.load,
                                             save=config.save,
                                             file_class="nn1",
                                             train_ratio=config.train_ratio,
                                             seq_len=config.max_len,
                                             min_len=config.min_len,
                                             step=-1)

    # Construct the data loader
    train_dataset = DatasetNN(train_data)
    valid_dataset = DatasetNN(val_data)
    train_data_loader = DataLoaderNN1(train_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      num_workers=0)
    valid_data_loader = DataLoaderNN1(valid_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      num_workers=0)

    # Initialize the evaluator NN
    # * Here we use a transformer model
    net = SampleNet(config)
    if torch.cuda.device_count() > 1:  # Multiple GPUs training
        net = nn.DataParallel(net)
    net.to(device)

    evaluator = Evaluator(config, net, device)
    torch.cuda.empty_cache()

    # Initialize the earlystopping agent
    early_stopping = EarlyStopping(patience=config.early_stop_patience,
                                   verbose=True)

    # Create the model store path
    model_store_path = config.model_store_path + config.dataset
    if not os.path.isdir(model_store_path):
        os.makedirs(model_store_path)

    # Start training the model
    if config.use_wandb == True:
        wandb.watch(evaluator.net,
                    evaluator.loss_function,
                    log="all",
                    log_freq=10)

    example_ct = 0
    print('Start training...\n')
    print("Using device: %s" % device)

    for epoch in range(config.epoch):
        # Train model
        evaluator.net.train()
        for i, (batch, _) in enumerate(train_data_loader):
            batch = batch.to(device)
            loss = evaluator.train_batch(batch)
            example_ct += len(batch)
            if config.use_wandb == True:
                wandb.log({"Epoch": epoch, "Loss": loss}, step=example_ct)

        # Validate model
        with torch.no_grad():
            evaluator.net.eval()
            losses = []
            for i, (batch, _) in enumerate(valid_data_loader):
                batch = batch.to(device)
                loss = evaluator.get_loss_on_eval_data(batch)
                losses.append(loss)

            avg_loss = sum(losses) / len(losses)
            print("Epoch: %i, Loss: %f" % (epoch, avg_loss))
            if config.use_wandb == True:
                wandb.log({
                    "Epoch": epoch,
                    "Val_loss": avg_loss
                },
                          step=example_ct)
            evaluator.pla_lr_scheduler.step(avg_loss)

            model_dict = {
                'epoch': epoch,
                'state_dict': evaluator.net.state_dict(),
                'optimizer': evaluator.optimizer.state_dict()
            }

            if epoch > 10:  # Todo: Hyperparameter: min_epoch
                model_name = '/eval_params.pth.tar' 
                early_stopping(avg_loss, model_dict, epoch,
                               model_store_path + model_name)
                if early_stopping.early_stop:
                    print("Early stopping...")
                    if config.use_wandb == True:
                        wandb.log(
                            {"val_loss_min": early_stopping.val_loss_min})
                    break

    print('Finish training...\n')


def load_evaluator(config, eval_config=None, device="cuda:0"):
    if eval_config == None:  # Called by nn1
        eval_config = config

    # Load model information
    model_store_path = config.model_store_path + config.dataset
    model_name = '/eval_params.pth.tar'  # if config.evaluator != "ngram" else '/ngram_params.pth.tar'
    model_info = torch.load(model_store_path + model_name)

    eval_config.n_item = config.n_item
    eval_config.n_user = config.n_user

    # Initialize the evaluator NN
    # * Here we use a transformer model
    net = SampleNet(eval_config)
    if torch.cuda.device_count() > 1:  # Multiple GPUs training
        net = nn.DataParallel(net)
    net.to(device)

    # Restore the evaluator
    net.load_state_dict(model_info['state_dict'])
    evaluator = Evaluator(eval_config, net, device)
    evaluator.optimizer.load_state_dict(model_info['optimizer'])
    return evaluator


def test_evaluator(config, dp, device="cuda:0"):
    # Load evaluator model
    evaluator = load_evaluator(config, None, device)
    torch.cuda.empty_cache()

    # Load testing data
    _, test_data = dp.get_refer_data(load=True,
                                     save=False,
                                     file_class="nn1",
                                     train_ratio=config.train_ratio,
                                     seq_len=config.max_len,
                                     min_len=config.min_len,
                                     step=-1)

    # Create testing dataset and dataloader
    test_dataset = DatasetNN(test_data)
    test_data_loader = DataLoaderNN1(test_dataset,
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     num_workers=0)

    # Start testing
    print("Testing evaluator...")
    print("Using device: %s" % device)
    with torch.no_grad():
        evaluator.eval()
        losses = []
        for i, (batch, _) in enumerate(test_data_loader):
            batch = batch.to(device)
            # Calculate loss
            loss = evaluator.get_loss_on_eval_data(batch)
            losses.append(loss)

        avg_loss = sum(losses) / len(losses)
        print("Number of test sequences: %i" % len(test_dataset))
        print("Test loss: %f" % avg_loss)
        if config.use_wandb == True:
            wandb.log({"uniform_loss": avg_loss})

    print("Finish testing...\n")
    
def eval_pipeline(config, device):
    # Load dataprovider
    dp = DataProvider(config, verbose = True)
    config.n_item = dp.n_item
    config.n_user = dp.n_user
    
    # Train nn1
    train_evaluator(config, dp, device)
    
    # Test nn1
    test_evaluator(config, dp, device)
