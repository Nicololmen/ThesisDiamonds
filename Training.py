# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:25:30 2022

@author: Milan

"""


import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import MLPNet
import CustomDataSet
from evaluation_metrics import calc_ap
import wandb
import argparse
from tqdm import tqdm
import numpy as np
import datetime


#ToDo betere functies

# passes batch through model and calculates cross_entropy loss
def run_train_step(model, batch, num_classes):
    inputs, labels = batch
    outputs = model(inputs)
    one_hot_targets = F.one_hot(labels, num_classes)
    return F.binary_cross_entropy_with_logits(
        outputs,
        one_hot_targets.float()
    )

# runs a training epoch
def run_train_epoch(model, train_loader, optimizer, epoch_idx, device='cpu'):
    model.train()
    num_classes = len(train_loader.dataset.classes)
    for batch_idx, train_batch in tqdm(enumerate(train_loader),
                                       total=len(train_loader),
                                       leave=False, desc='Train batch'):
        train_batch = batch_to_device(train_batch, device)
        loss = run_train_step(model, train_batch, num_classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log the training loss
        wandb.log({"epoch": epoch_idx,
            "train_loss": loss})

# moves batch to device
def batch_to_device(batch, device):
    batch[0] = batch[0].to(device)
    batch[1] = batch[1].to(device)

    return batch


#
def run_val_epoch(model, val_loader, epoch_idx, writer, device='cpu'):
    # Put model in eval mode
    model.eval()

    logits, q_labels = compute_logits_from_dataloader(
        model,
        val_loader,
        device
    )

    # Compute similarity matrix by applying softmax to logits
    sim_mat = F.softmax(logits, dim=1)

    # Log average precision
    idx_to_class = {
        idx: class_name
        for class_name, idx in val_loader.dataset.class_to_idx.items()
    }

    # Create an array with the labels (indices) in the dataset
    uniq_labels = np.array(list(idx_to_class))

    for label in uniq_labels:
        ap = calc_ap(label, sim_mat, uniq_labels, q_labels)
        wandb.log({"epoch": epoch_idx,
                   "avarage_precision": ap})


def compute_logits_from_dataloader(model, data_loader, device='cpu'):
    all_logits = []
    all_labels = []

    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
        batch = batch_to_device(batch, device)
        imgs, labels = batch

        with torch.no_grad():
            logits = model(imgs)

        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits).cpu()
    all_labels = torch.cat(all_labels).cpu().numpy()

    return all_logits, all_labels


def run_training(model, optimizer, train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    for epoch_idx in tqdm(range(num_epochs), desc='Epoch'):
        run_train_epoch(model, train_loader, optimizer, epoch_idx, device)
        run_val_epoch(model, val_loader, epoch_idx, device)

def run_training(model, optimizer, train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)



    for epoch_idx in tqdm(range(num_epochs), desc='Epoch'):
        run_train_epoch(model, train_loader, optimizer, epoch_idx, device)
        run_val_epoch(model, val_loader, epoch_idx, device)

def main():
    
    #Get Flags
    parser=argparse.ArgumentParser(description='Flags for traing hyper parameters')
    
    parser.add_argument("--data_set", help='Supply path to dataset, default value=NormalizedRough[D-I].csv.', default="E:\\DatasetDiamonds\\NormalizedPolished[D-I].csv")
    parser.add_argument("--learning_rate", help='Supply learning rate for training, default value=0.001', default='0.001')
    parser.add_argument("--weight_decay", help='Supply weight decay for training, default value=0.0001', default='0.0001')
    parser.add_argument("--epochs", help='Supply amount of epochs for training, default value=25', default='25')
    parser.add_argument("--batch_size", help='Supply batch_size for training loader, default value=32', default='32')
    parser.add_argument("--num_workers", help='Supply num_workers for training loader, default value=1', default='1')
    parser.add_argument("--num_nodes_per_layer", help="Give model architecture eg 512-256-128-6", default="512-256-64-6")

    
    args=parser.parse_args()

    # Get Headers
    Data = pd.read_csv(args.dataset, skiprows=0, nrows=2)
    Headers = Data.columns

    # Get training Data
    data_set = CustomDataSet.CustomDataSet(args.data_set, 1, 9000, Headers)
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))

    # Get test data
    val_data_set = CustomDataSet.CustomDataSet(args.data_set, 9000, 10428, Headers)
    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))

    #set-up weight and biases
    wandb.require(experiment="service")
    wandb.init(name=f'training_run {args.num_nodes_per_layer} {datetime.datetime.now().date()}',
               project='Thesis Diamonds')
    wandb.config.lr=float(args.learning_rate)
    wandb.config.wd=float(args.weight_decay)
    

    #make the network model
    model = MLPNet.Net(args.num_nodes_per_layer)
    
    #Initialize learning parameters

    optimizer = optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    run_training(model, optimizer, train_loader,
                 val_loader, num_epochs=25)


if __name__ == "__main__":
    main()