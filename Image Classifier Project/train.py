#!/usr/bin/env python
# PROGRAMMER: Zachary Amerman
# DATE CREATED: April 23, 2019
# REVISED DATE: April 24, 2019
# PURPOSE: Creates, trains, validates, and saves a neural network.

# Import pytorch libraries
import torch.nn as nn

# Import custom libraries
from get_input_args_T import get_input_args
from get_dataloaders import get_dataloaders
from build_model import build_model
from train_model import train_model
from save_checkpoint import save_checkpoint

def main():
    # Get command line arguments
    args = get_input_args()
    
    # Get training and validation dataloaders
    dataloaders, class_to_idx = get_dataloaders(args.data_dir)
    
    # Build network model and optimizer
    model, optimizer = build_model(args.arch, args.hidden_units, args.lr)
                
    # Set optimizers and criterion
    
    criterion = nn.NLLLoss()
    
    # Train and validate model
    train_model(model, optimizer, criterion, args.epochs, dataloaders['train'], dataloaders['valid'], args.device)
    
    # Save model
    save_checkpoint(model, args.arch, args.hidden_units, args.epochs, criterion, optimizer, args.save_dir, class_to_idx, args.lr)


if __name__ == "__main__":
    main()