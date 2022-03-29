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
from HulpFunctions.evaluation_metrics import calculate_accuracy
import wandb
import argparse
from tqdm import tqdm
import numpy as np
import datetime


#
def run_val_epoch(model, val_loader, epoch_index, device='cpu'):
    # Put model in eval mode
    model.eval()

    labels, predictions = run_eval_step(model, val_loader, device)

    accuracy = calculate_accuracy(torch.Tensor(labels), torch.Tensor(predictions))

    # Log the evaluation accuracy
    wandb.log({"epoch": epoch_index,
               "evaluation accuracy": accuracy})


def run_eval_step(model, data_loader, device='cpu'):
    all_logits = []
    all_labels = []

    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
        batch = batch_to_device(batch, device)
        inputs, labels = batch

        with torch.no_grad():
            logits = model(inputs)

        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits).cpu()
    all_labels = torch.cat(all_labels).cpu().numpy()

    return all_logits, all_labels

# passes batch through model and calculates and returns cross_entropy loss
def run_train_step(model, batch, num_classes):
    inputs, labels = batch
    outputs = model(inputs)
    one_hot_targets = F.one_hot(labels, num_classes)
    print(one_hot_targets)
    return F.binary_cross_entropy_with_logits(
        outputs,
        one_hot_targets.float()
    )


# moves batch to device
def batch_to_device(batch, device):
    batch[0] = batch[0].to(device)
    batch[1] = batch[1].to(device)

    return batch

def run_train_epoch(model, train_loader, optimizer, epoch_index, device='cpu'):
    model.train()
    num_classes = 6 #len(train_loader.dataset.classes)   Dit werkt niet weet niet goed vanwaar dit komt? extra veldje in CustomDataSet klasse?
    for batch_index, train_batch in tqdm(enumerate(train_loader),
                                       total=len(train_loader),
                                       leave=False, desc='Train batch'):
        train_batch = batch_to_device(train_batch, device)
        loss = run_train_step(model, train_batch, num_classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log the training loss
        wandb.log({"epoch": epoch_index,
            "training loss": loss})


def run_training(model, optimizer, train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch_index in tqdm(range(num_epochs), desc='Epoch'):
        run_train_epoch(model, train_loader, optimizer, epoch_index, device)
       #run_val_epoch(model, val_loader, epoch_index, device)

def main():
    
    #Get Flags
    parser=argparse.ArgumentParser(description='Flags for training hyper parameters')
    
    parser.add_argument("--data_set", help='Supply path to dataset, default value=NormalizedRough[D-I].csv.', default="E:\\DatasetDiamonds\\NormalizedRough[D-I].csv")
    parser.add_argument("--learning_rate", help='Supply learning rate for training, default value=0.001', default='0.001')
    parser.add_argument("--weight_decay", help='Supply weight decay for training, default value=0.0001', default='0.0001')
    parser.add_argument("--epochs", help='Supply amount of epochs for training, default value=25', default='25')
    parser.add_argument("--batch_size", help='Supply batch_size for training loader, default value=32', default='32')
    parser.add_argument("--num_workers", help='Supply num_workers for training loader, default value=1', default='1')
    parser.add_argument("--num_nodes_per_layer", help="Give model architecture eg 512-256-128-6", default="512-256-64-6")
    parser.add_argument("--P_or_R", help="1 for polished 0 for rough", default="0")

    args=parser.parse_args()

    # Get Headers
    Data = pd.read_csv(args.data_set, skiprows=0, nrows=2)
    Headers = Data.columns
    if(int(args.P_or_R)):
        drop_headers = ['Lot', 'SCTF_CLARITY', 'PolishedPicture', 'IntegrationTimePolished', 'SCTF_FLUO', 'GIAColor', 'FileName', 'PolishedFile', 'wp']
    if(not(int(args.P_or_R))):
        drop_headers = ['Lot', 'MR_CLARITY', 'RoughPicture', 'IntegrationTimeRough', 'MR_FLUO', 'GIAColor', 'FileName', 'wp', 'wr']

    # Get training Data
    data_set = CustomDataSet.CustomDataSet(args.data_set, 1, 9000, Headers, drop_headers)
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    # Get test data
    val_data_set = CustomDataSet.CustomDataSet(args.data_set, 9000, 10428, Headers, drop_headers)
    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))

    # Set-up weight and biases
    wandb.require(experiment="service")
    wandb.init(name=f'training_run {args.num_nodes_per_layer} {datetime.datetime.now().date()}',
               project='Thesis Diamonds')
    wandb.config.lr=float(args.learning_rate)
    wandb.config.wd=float(args.weight_decay)
    

    # Make the network model
    model = MLPNet.Net(args.num_nodes_per_layer)
    
    # Initialize learning parameters
    optimizer = optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    # Run model training
    run_training(model, optimizer, train_loader, val_loader, num_epochs=int(args.epochs))


if __name__ == "__main__":
    main()