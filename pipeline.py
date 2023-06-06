# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:57:00 2021

@author: ZHU Haoren
"""
import torch
import torch.nn as nn
import numpy as np
import os
import wandb
import argparse

from data_provider import DataProvider, DataLoaderEvalIRS, DatasetEvalNN1, DataLoaderEvalNN1, DatasetNN, DataLoaderIRS
from model.influentialRS import IRSNN, InfluentialNet
from utils import EarlyStopping
from evaluator_pipeline import load_evaluator


def train_model(config, dp, device):
    # Load training data and validation data
    train_data, val_data = dp.get_refer_data(load=config.load,
                                             save=config.save,
                                             file_class="irs",
                                             train_ratio=config.train_ratio,
                                             seq_len=config.max_len,
                                             min_len=config.min_len,
                                             step=config.step)
    # Generate dataset and dataloader
    train_dataset = DatasetNN(train_data)
    valid_dataset = DatasetNN(val_data)
    train_data_loader = DataLoaderIRS(train_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      num_workers=0)
    valid_data_loader = DataLoaderIRS(valid_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      num_workers=0)

    # Create core model allowing for multiple GPU training
    net = InfluentialNet(config)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    # Create handler
    iRSNN = IRSNN(config, net, device)

    torch.cuda.empty_cache()

    # Early stopping handler
    model_store_path = config.model_store_path + config.dataset
    if not os.path.isdir(model_store_path):
        os.makedirs(model_store_path)
    early_stopping = EarlyStopping(patience=config.early_stop_patience,
                                   verbose=True)

    # Loss function
    loss_function = nn.CrossEntropyLoss()

    # Use wandb to track model
    if config.use_wandb == True:
        wandb.watch(iRSNN, loss_function, log="all", log_freq=10)
    example_ct = 0

    # Start training the model
    print('Start training...\n')
    print("Using device: %s" % device)
    for epoch in range(config.epoch):

        # Train model
        iRSNN.net.train()
        for i, (seqs, users) in enumerate(train_data_loader):

            seqs = seqs.to(device)
            users = users.to(device)
            loss = iRSNN.train_batch(seqs, users)

            # Update information to wandb
            example_ct += len(seqs)
            if config.use_wandb == True:
                wandb.log({"Epoch": epoch, "Loss": loss}, step=example_ct)

        # Validate model
        with torch.no_grad():
            iRSNN.net.eval()
            losses = []
            for i, (seqs, users) in enumerate(valid_data_loader):
                seqs = seqs.to(device)
                users = users.to(device)
                loss = iRSNN.get_loss_on_eval_data(seqs, users)
                losses.append(loss)

            # Calculate the average loss over the validation set
            avg_loss = sum(losses) / len(losses)
            print("Epoch: %i, Loss: %f" % (epoch, avg_loss))

            if config.use_wandb == True:
                wandb.log({
                    "Epoch": epoch,
                    "Val_loss": avg_loss
                },
                          step=example_ct)

            # Step the learning rate scheduler
            iRSNN.pla_lr_scheduler.step(avg_loss)

            # Store the model
            model_dict = {
                'epoch': epoch,
                'state_dict': iRSNN.net.state_dict(),
                'optimizer': iRSNN.optimizer.state_dict()
            }
            if epoch > 10:  # Todo: Hyperparameter: min_epoch
                early_stopping(avg_loss, model_dict, epoch,
                               model_store_path + '/irn_params.pth.tar')

                if early_stopping.early_stop:
                    print("Early stopping...")
                    if config.use_wandb == True:
                        wandb.log(
                            {"val_loss_min": early_stopping.val_loss_min})
                    break

    print('Finish training...\n')


def load_model(config, device="cuda:0"):
    """
    Load existing nn1
    """
    # Load model information
    model_store_path = config.model_store_path + config.dataset
    model_info = torch.load(model_store_path + '/irn_params.pth.tar')

    # Restore the core module
    net = InfluentialNet(config)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)
    net.load_state_dict(model_info['state_dict'])

    # Restore the handler
    irn = IRSNN(config, net, device)
    irn.optimizer.load_state_dict(model_info['optimizer'])
    return irn


def test_model(config, dp, device="cuda:0"):
    print("Testing nn1...")
    print("Using device: %s" % device)
    irn = load_model(config, device)

    # Save the generated path results
    result_save_dir = "./results/" + config.dataset + '/' + config.method + '/'
    if not os.path.isdir(result_save_dir):
        os.makedirs(result_save_dir)

    eval_data = dp.get_random_evaluate_data(seq_len=100,
                                            use_train=config.use_train,
                                            file_class="irs",
                                            gap_len=config.gap_len)

    # Random Target Task
    print("Evaluate with random target...\n")
    eval_dataset = DatasetNN(eval_data)
    eval_data_loader = DataLoaderEvalIRS(dataset=eval_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=False,
                                         num_workers=0,
                                         gap_len=config.gap_len,
                                         seq_len=config.max_len)
    n_eval = len(eval_dataset)

    with torch.no_grad():
        irn.eval()
        paths = []  # The persuasion path
        n_early_success = 0  # The number of early success
        histories = []  # The input sequence
        targets = []  # The target item
        reverse_ranks = []  # MRR
        r_us = []  # Personalized factor
        hit = 0  # Hit Rate

        for i, (raw, seq, u, t, l) in enumerate(eval_data_loader):
            # Move tensors to device
            t = t.to(device)  # Target
            l = l.to(device)  # Label
            u = u.to(device)  # User
            seq = seq.to(device)  # Sequence

            # Obtain the r_u
            r_u = irn.get_pif_in_batch(seq, u)

            # Append the result
            if len(r_us) == 0:
                r_us = r_u
            else:
                r_us = np.concatenate((r_us, r_u))

            # Obtain accuracy results
            hit_count, rr = irn.get_accuracy_metrics_in_batch(
                raw, seq, u, t, l, config.top_k, config.gap_len, config.use_h)

            # Append the batch result to all results
            hit += hit_count  # increment hit counter
            if len(reverse_ranks) == 0:  # first batch
                reverse_ranks = rr
            else:  # The following batch
                reverse_ranks = np.concatenate([reverse_ranks, rr])

            # Generate the recommendation path
            p, t, h, early_success = irn.get_seq_in_batch(
                seq, u, t, config.max_path_len, config.gap_len, config.sample,
                config.sample_k)

            # Append the batch result to all results
            n_early_success += early_success
            if len(paths) == 0:
                paths = p
                histories = h
                targets = t
            else:
                paths = np.vstack((paths, p))
                histories += h
                targets = np.concatenate((targets, t))

        # Calculate the accuracy results
        hit = hit / n_eval
        mrr = sum(reverse_ranks) / n_eval
        print("Hit rate: %f" % hit)
        print("MRR: %f" % mrr)
        print("Early Success: %i" % n_early_success)
        print("Early Success Rate: %f" % (n_early_success / n_eval))

        if config.use_wandb == True:
            wandb.log({"hit": hit, "sr": n_early_success / n_eval})

        # Save the input and the derived path
        np.save(result_save_dir + "paths_d_%i.npy" % config.max_path_len,
                paths)
        np.save(result_save_dir + "histories.npy", histories)
        np.save(result_save_dir + "targets_d_%i.npy" % config.max_path_len,
                targets)
        np.save(result_save_dir + "r_u.npy", r_us)


def evaluate_prob(config, eval_config, dp, device="cuda:0"):
    """
    Evaluate the result with probability related metrics.
    The probability is generated from an evaluator
    """
    print("Evaluating the path...")
    print("Using device: %s" % device)
    eval_config = argparse.Namespace(**eval_config)
    evaluator = load_evaluator(config, eval_config, device)

    result_save_dir = "./results/" + config.dataset + '/' + config.method + '/'
    histories = np.load(result_save_dir + "histories.npy", allow_pickle=True)
    targets = np.load(result_save_dir +
                      "targets_d_%i.npy" % config.max_path_len,
                      allow_pickle=True)
    paths = np.load(result_save_dir + "paths_d_%i.npy" % config.max_path_len,
                    allow_pickle=True)

    eval_dataset = DatasetEvalNN1(histories,
                                  paths,
                                  targets,
                                  seq_len=eval_config.max_len)
    eval_data_loader = DataLoaderEvalNN1(eval_dataset,
                                         batch_size=eval_config.batch_size,
                                         shuffle=False,
                                         num_workers=0)

    with torch.no_grad():
        evaluator.eval()
        perplexity = []  # Perplexity
        p_probs = []  # Path item prob
        t_probs = []  # Target prob
        iis = []  # Increase of interest
        irrs = []  # Increase of reverse ranking
        irs = []  # Increase of ranking
        avg_acpt = []  # average accpetance prob

        for i, (h, d, t, sp, lp) in enumerate(eval_data_loader):
            h = h.to(device)
            d = d.to(device)
            t = t.to(device)
            sp = sp.to(device)
            lp = lp.to(device)

            # Calculate perplexity
            pp = evaluator.get_pp_in_batch(d, sp, lp)
            # Calculate the increase of reverse ranking/ranking
            irr, ir = evaluator.get_rr_increase_in_batch(h, d, t)
            # Calculate probability related metrics
            target_ps, path_ps, avg_ps, ii = evaluator.get_grad_in_batch(
                h, d, t, sp, lp)

            if len(perplexity) == 0:
                perplexity = pp
                iis = ii
                irrs = irr
                irs = ir
                avg_acpt = avg_ps
                p_probs = path_ps
                t_probs = target_ps
            else:
                perplexity = np.concatenate((perplexity, pp))
                iis = np.concatenate((iis, ii))
                irrs = np.concatenate((irrs, irr))
                irs = np.concatenate((irs, ir))
                avg_acpt = np.concatenate((avg_acpt, avg_ps))
                p_probs = np.vstack((p_probs, path_ps))
                t_probs = np.vstack((t_probs, target_ps))

        length = len(perplexity)
        perplexity = sum(perplexity) / length
        ii = sum(iis) / length
        irr = sum(irrs) / length
        ir = sum(irs) / length
        ap = sum(avg_acpt) / length
        print("The perplexity: %f" % perplexity)
        print("The increase of interest: %f" % ii)
        print("The increase of reverse ranking: %f" % irr)
        print("The increase of ranking: %f" % ir)
        print("The average acceptance probability: %f" % ap)
        np.save(result_save_dir + "p_probs.npy", p_probs)
        np.save(result_save_dir + "t_probs.npy", t_probs)


def pipeline(config, evaluator_config, device='cpu'):
    dp = DataProvider(config, verbose=True)
    config.n_item = dp.n_item
    config.n_user = dp.n_user

    # Train nn1
    if config.is_train == True:
        train_model(config, dp, device)

    # Test nn1
    if config.is_test == True:
        test_model(config, dp, device)

    # Evaluate the path
    if config.is_eval == True:
        evaluate_prob(config, evaluator_config, dp, device)
