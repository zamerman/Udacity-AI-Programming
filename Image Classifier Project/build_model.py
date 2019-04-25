#!/usr/bin/env python
# PROGRAMMER: Zachary Amerman
# DATE CREATED: April 23, 2019
# REVISED DATE:
# PURPOSE: Downloads a pretrained neural network and then adds a custom
#       classifier to it with a variable number of hidden units.

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Define classifier model with one hidden layer with relu activations and dropout
class Classifier(nn.Module):
    def __init__(self, inputs, hidden_units):
        super().__init__()
        
        self.hidden = nn.Linear(inputs, hidden_units)
        self.output = nn.Linear(hidden_units, 102)
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        
        return x
    
def build_model(arch, hidden_units):
    """
    Builds a neural network from a pretrained architecture with a
    variable number of hidden_units
    Parameters
     arch - a shorhand string for an architecture
     hidden_units - the number of hidden units to include in the classifier
    Returns
     model - a nn.Module neural network
    """
    
    # Select our base pretrained architecture
    if arch == "vgg11":
        model = models.vgg11(pretrained=True)
        inputs = 25088
    if arch == "alexnet":
        model = models.alexnet(pretrained=True)
        inputs = 9216
    if arch == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=True)
        inputs = 512
    if arch == "densenet121":
        model = models.densenet121(pretrained=True)
        inputs = 1024
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifiers
    model.classifier = Classifier(inputs, hidden_units)
    
    return model