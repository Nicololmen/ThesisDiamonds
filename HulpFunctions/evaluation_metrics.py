#Hierin staan functies die gebruikt kunnen worden om het model te evalueren

import torch



def calculate_accuracy(labels, predictions):
    #This function returns the accuracy tp/number_predictions
    #The function takes 2 tensors
    #labels --> ground truth, predictions --> the raw output of the neural network model
    _, predicted = torch.max(predictions.data, 1)
    return ((predicted == labels).sum().item()/labels.size(0))*100

