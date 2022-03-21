# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:25:30 2022

@author: Milan

"""


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import MLPNet
import CustomDataSet
import wandb
import argparse

#ToDo betere functies
def calcTop1Accuracy(Data_loader, net, device, criterion, doif=False, wandb=None):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in Data_loader:
            spectras, label = data[0].to(device), data[1].to(device)   #voor gpu
            
            outputs = net(spectras)
            loss=criterion(outputs, label)
          
            _, predicted = torch.max(outputs.data, 1)
            
            if(doif):
              wandb.log({"PR" : wandb.plot.pr_curve(label, 
                         outputs, labels=['D', 'E', 'F', 'G', 'H', 'I'])})
            total += label.size(0)
            correct += (predicted == label).sum().item()
        acc=100 * correct / total


    return acc, loss

def main():
    
    #Get Flags
    parser=argparse.ArgumentParser(description='Flags for traing hyper parameters')
    
    parser.add_argument("--dataset", help='Supply path to dataset, default value=NormalizedRough[D-I].csv.', default="E:\\DatasetDiamonds\\NormalizedPolished[D-I].csv")
    parser.add_argument("--learningRate", help='Supply learning rate for training, default value=0.001', default='0.001')
    parser.add_argument("--weightDecay", help='Supply weight decay for training, default value=0.0001', default='0.0001')
    parser.add_argument("--epochs", help='Supply amount of epochs for training, default value=25', default='25')
    parser.add_argument("--batchSize", help='Supply batch_size for training loader, default value=32', default='32')
    parser.add_argument("--numNodesPerLayer", help="", default="512-256-64-6")
    
    args=parser.parse_args()
    
    #set-up weight and biases
    
    wandb.require(experiment="service")
    wandb.init(name=f'training_run {}',   #ToDo: nieuwe naam die uniek is
               project='Thesis Diamonds')
    wandb.config.lr=float(args.learningRate)
    wandb.config.wd=float(args.weightDecay)
    
    #Get Headers
    Data=pd.read_csv(args.dataset, skiprows=0, nrows=2)
    Headers=Data.columns
    
    #Get training Data
    Dataset=CustomDataSet.CustomDataSet(args.dataset, 1, 9000, Headers)
    train_loader=torch.utils.data.DataLoader(Dataset, batch_size=int(args.batchSize), shuffle=True , num_workers=1)
    
    #Get test data
    TestDataset=CustomDataSet.CustomDataSet(args.dataset, 9000, 10428, Headers)
    test_loader=torch.utils.data.DataLoader(TestDataset, batch_size=int(args.batchSize), shuffle=True , num_workers=1)

    TestDatasetD=CustomDataSet.CustomDataSet("E:\\DatasetDiamonds\\NormalizedPolished[D].csv", 1, 219, Headers)
    test_loaderD=torch.utils.data.DataLoader(TestDatasetD, batch_size=int(args.batchSize), shuffle=True , num_workers=1)

    TestDatasetE=CustomDataSet.CustomDataSet("E:\\DatasetDiamonds\\NormalizedPolished[E].csv", 1, 251, Headers)
    test_loaderE=torch.utils.data.DataLoader(TestDatasetE, batch_size=int(args.batchSize), shuffle=True , num_workers=1)

    TestDatasetF=CustomDataSet.CustomDataSet("E:\\DatasetDiamonds\\NormalizedPolished[F].csv", 1, 308, Headers)
    test_loaderF=torch.utils.data.DataLoader(TestDatasetF, batch_size=int(args.batchSize), shuffle=True , num_workers=1)

    TestDatasetG=CustomDataSet.CustomDataSet("E:\\DatasetDiamonds\\NormalizedPolished[G].csv", 1, 266 , Headers)
    test_loaderG=torch.utils.data.DataLoader(TestDatasetG, batch_size=int(args.batchSize), shuffle=True , num_workers=1)

    TestDatasetH=CustomDataSet.CustomDataSet("E:\\DatasetDiamonds\\NormalizedPolished[H].csv", 1, 224 , Headers)
    test_loaderH=torch.utils.data.DataLoader(TestDatasetH, batch_size=int(args.batchSize), shuffle=True , num_workers=1)

    TestDatasetI=CustomDataSet.CustomDataSet("E:\\DatasetDiamonds\\NormalizedPolished[I].csv", 1, 161, Headers)
    test_loaderI=torch.utils.data.DataLoader(TestDatasetI, batch_size=int(args.batchSize), shuffle=True , num_workers=1)

    #make the network model
    net=MLPNet.Net()
    
    #Initialize learning parameters
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(net.parameters(), lr=float(args.learningRate), weight_decay=float(args.weightDecay))
    
    #Load network to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    #monitor network
    wandb.watch(net, criterion)
    
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
     
      testacc, testloss=calcTop1Accuracy(test_loader, net, device, criterion, True, wandb)
      testaccD, testlossD=calcTop1Accuracy(test_loaderD, net, device, criterion)
      testaccE, testlossE=calcTop1Accuracy(test_loaderE, net, device, criterion)
      testaccF, testlossF=calcTop1Accuracy(test_loaderF, net, device, criterion)
      testaccG, testlossG=calcTop1Accuracy(test_loaderG, net, device, criterion)
      testaccH, testlossH=calcTop1Accuracy(test_loaderH, net, device, criterion)
      testaccI, testlossI=calcTop1Accuracy(test_loaderI, net, device, criterion)
      trainacc, trainloss=calcTop1Accuracy(train_loader, net, device, criterion)
        
        
        
      wandb.log({"epoch": (epoch+1),
                "train_loss": trainloss,
                "train_acc": trainacc,
                "test_loss": testloss,
                "test_acc": testacc,
                "test_acc_D": testaccD,
                "test_loss_D": testlossD,
                "test_acc_E": testaccE,
                "test_loss_E": testlossE,
                "test_acc_F": testaccF,
                "test_loss_F": testlossF,
                "test_acc_G": testaccG,
                "test_loss_G": testlossG,
                "test_acc_H": testaccH,
                "test_loss_H": testlossH,
                "test_acc_I": testaccI,
                "test_loss_I": testlossI
                })

if __name__ == "__main__":
    main()