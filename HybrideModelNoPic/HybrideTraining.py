# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:25:30 2022

@author: Milan

"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import HybrideNet
import HybrideDataSet
from HulpFunctions.evaluation_metrics import calc_ap
import wandb
import argparse
from tqdm import tqdm
import numpy as np
import datetime
import os


#
def run_val_epoch(model, val_loader, epoch_index, device='cpu'):
    # Put model in evaluation mode
    model.eval()

    predictions, labels = run_eval_step(model, val_loader, device)

    sim_mat = F.softmax(predictions, dim=1)

    uniq_labels = np.array([0, 1, 2, 3, 4, 5])

    for label in uniq_labels:
        try:
            ap = calc_ap(label, sim_mat, uniq_labels, labels)
            # Log the evaluation accuracy
            wandb.log({"epoch": epoch_index + 1,
                       "avarage precision": ap}
                       )
        except:
            print(f"Geen records gevonden voor {label}")


def run_eval_step(model, data_loader, device='cpu'):
    all_logits = []
    all_labels = []

    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
        batch = batch_to_device(batch, device)
        features, z, labels = batch

        with torch.no_grad():
            logits = model(features, z)

        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits).cpu()
    all_labels = torch.cat(all_labels).cpu().numpy()

    return all_logits, all_labels


# passes batch through model and calculates and returns cross_entropy loss
def run_train_step(model, batch, num_classes):
    features, z, labels = batch

    outputs = model(features, z)

    one_hot_targets = F.one_hot(labels, num_classes)
    return F.binary_cross_entropy_with_logits(
        outputs,
        one_hot_targets.float()
    )


# moves batch to device
def batch_to_device(batch, device):
    batch[0] = batch[0].to(device)
    batch[1] = batch[1].to(device)
    batch[2] = batch[2].to(device)
    

    return batch


def run_train_epoch(model, train_loader, optimizer, epoch_index, device='cpu'):
    model.train()
    num_classes = 6  # len(train_loader.dataset.classes)   Dit werkt niet weet niet goed vanwaar dit komt? extra veldje in CustomDataSet klasse?
    for batch_index, train_batch in tqdm(enumerate(train_loader),
                                         total=len(train_loader),
                                         leave=False, desc='Train batch'):
        train_batch = batch_to_device(train_batch, device)
        loss = run_train_step(model, train_batch, num_classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log the training loss
        wandb.log({"epoch": epoch_index + 1,
                   "training loss": loss})


def run_training(model, optimizer, train_loader, val_loader, scheduler, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch_index in tqdm(range(num_epochs), desc='Epoch'):
        run_train_epoch(model, train_loader, optimizer, epoch_index, device)
        run_val_epoch(model, val_loader, epoch_index, device)
        torch.save(model.state_dict(), os.path.join(r"C:\Users\Milan\Desktop\Training", str(epoch_index) + ".pt"))
        scheduler.step()


def main():
    # Meta data
    data_sets = [r"E:\\DatasetDiamonds\\NormalizedRough[D-I].csv", r"E:\\DatasetDiamonds\\NormalizedPolished[D-I].csv"]

    # Get Flags
    parser = argparse.ArgumentParser(description='Flags for training hyper parameters')

    parser.add_argument("--data_set", help='Supply 0 for rough and 1 for polished default value=0', default="0")
    parser.add_argument("--learning_rate", help='Supply learning rate for training, default value=0.001',
                        default='0.001')
    parser.add_argument("--learning_rate_decay_step", help="Supply step (in epochs) for lr decay", default="8")
    parser.add_argument("--learning_rate_gamma", help="Supply gamma vor lr decay default value=0.1", default="0.3")
    parser.add_argument("--weight_decay", help='Supply weight decay for training, default value=0.0001',
                        default='0.0001')
    parser.add_argument("--epochs", help='Supply amount of epochs for training, default value=25', default='15')
    parser.add_argument("--batch_size", help='Supply batch_size for training loader, default value=32', default='128')
    parser.add_argument("--num_workers", help='Supply num_workers for training loader, default value=1', default='0')
    parser.add_argument("--num_nodes_per_layer", help="Give model architecture eg 43-512-256-128-6",
                        default="34-32-6")


    args = parser.parse_args()

    # set up wandb and get config
    wandb.require(experiment="service")
    wandb.init(config=args,
               name=f"training_run  wp, ITR, FLUO, CLAR date={datetime.datetime.now().date()} dataset={data_sets[int(args.data_set)]}",
               project="HybrideModelAblation")
    config = wandb.config

    # Get Headers
    Data = pd.read_csv(data_sets[int(config["data_set"])], skiprows=0, nrows=2)
    Headers = Data.columns
    if (int(config["data_set"])):
        drop_headers = ['PolishedPicture', 'GIAColor', 'FileName', 'PolishedFile']
    if (not (int(config["data_set"]))):
        drop_headers =  ['RoughPicture','GIAColor', 'FileName']
    # Get training Data
    data_set = HybrideDataSet.HybrideDataSet(data_sets[int(config["data_set"])], 1, 8000, Headers, drop_headers)
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=int(config["batch_size"]), shuffle=True,
                                               num_workers=int(config["num_workers"]))
    # Get test data
    val_data_set = HybrideDataSet.HybrideDataSet(data_sets[int(config["data_set"])], 8000, 10428, Headers, drop_headers)
    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=int(config["batch_size"]), shuffle=True,
                                             num_workers=int(config["num_workers"]))

    # Make the network model

    model = HybrideNet.HybrideNet(config["num_nodes_per_layer"])


    # Initialize learning parameters
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]),
                           weight_decay=float(config["weight_decay"]))

    # Learning rate decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(config['learning_rate_decay_step']),
                                          gamma=float(config['learning_rate_gamma']))

    # Run model training
    run_training(model, optimizer, train_loader, val_loader, scheduler, num_epochs=int(config["epochs"]))

    

if __name__ == "__main__":
    main()