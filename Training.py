# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:25:30 2022

@author: Milan

"""


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import MLPNet.py
import CustomDataSet.py
import wandb
import argparse

#calculates the top 1 accuracy --> tp/(len(data_loader)) and loss
def calcTop1Accuracy(Data_loader, net, device, criterion):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in Data_loader:
            spectras, labels = data[0].to(device), data[1].to(device)   #voor gpu
          
            outputs = net(spectras)
            loss=criterion(outputs, labels)
          
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc=100 * correct / total
    return acc, loss

def main():
    
    #Get Flags
    parser=argparse.ArgumentParser(description='Flags for traing hyper parameters')
    
    parser.add_argument("--dataset", help='Supply path to dataset, default value=NormalizedRough[D-I].csv.', default='NormalizedRough[D-I].csv')
    parser.add_argument("--learningRate", help='Supply learning rate for training, default value=0.001', default='0.001')
    parser.add_argument("--weightDecay", help='Supply weight decay for training, default value=0.0001', default='0.0001')
    parser.add_argument("--epochs", help='Supply amount of epochs for training, default value=25', default='25')
    parser.add_argument("--batchSize", help='Supply batch_size for training loader, default value=32', default='32')
    
    args=parser.pars_args()
    
    #set-up weight and biases
    wandb.login()
    wandb.init(name='training_run',
               project='Thesis Diamonds',
               entity='nicololmen')
    wandb.config.lr=int(args.learningRate)
    wandb.config.wd=int(args.weightDecay)
    
    #Get Headers
    Data=pd.read_csv(args.dataset, skiprows=0, nrows=2)
    Headers=Data.columns
    
    #Get training Data
    Dataset=CustomDataSet.CustomDataSet(args.dataset, 1, 9000, Headers)
    train_loader=torch.utils.data.DataLoader(Dataset, batch_size=int(args.batchSize), shuffle=True , num_workers=1)
    
    #Get test data
    TestDataset=CustomDataSet.CostumDataSet(args.dataset, 9000, 10428, Headers)
    test_loader=torch.utils.data.DataLoader(TestDataset, batch_size=int(args.batchSize), shuffle=True , num_workers=1)
    
    #make the network model
    net=MLPNet.Net()
    
    #Initialize learning parameters
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(net.parameters(), lr=int(args.learningRate), weight_decay=int(args.weightDecay))
    
    #Load network to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    #monitor network
    wandb.watch(net)
    
    #train loop
    for epoch in range(int(args.epochs)):
      net.train()
      
      running_loss=0.0
      for i, data in enumerate(train_loader):
        #inputs, labels=data  #voor cpu 
        inputs, labels = data[0].to(device), data[1].to(device)   #voor gpu
        
        outputs=net(inputs)
        
        loss=criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
    
        optimizer.step()
    
        running_loss+=loss.item()
      if((epoch+1)%5==0 and epoch!=0):
        testacc, testloss=calcTop1Accuracy(test_loader, net, device, criterion)
        trainacc, trainloss=calcTop1Accuracy(train_loader, net, device, criterion)
        
        print('After %d epochs train loss: %f, train accuracy: %d%%, test loss: %f test accuracy: %d%%' % (epoch+1, trainloss, trainacc, testloss, testacc))
        
        wandb.log({"epoch": (epoch+1),
                  "train_loss": trainloss,
                  "train_acc": trainacc,
                  "test_loss": testloss,
                  "test_acc": testacc})

main()