# Import torch libraries
import torch
import torch.nn as nn
from torch import optim

# Import custom function for building models
from build_model import build_model

def load_checkpoint(path):
    # Load checkpoint
    checkpoint = torch.load(path)
    
    # Create model and copy over loaded state_dict create optimizer to go with model
    model, optimizer = build_model(checkpoint['arch'], checkpoint['hidden_units'], checkpoint['lr'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optim_state'])
    
    # Load epochs
    epochs = checkpoint['epochs']
    
    # Load criterion
    criterion = nn.NLLLoss()
    criterion.load_state_dict(checkpoint['crit_state'])
    
    return model, epochs, criterion, optimizer