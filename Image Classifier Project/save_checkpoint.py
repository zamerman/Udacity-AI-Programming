#!/usr/bin/env python
# PROGRAMMER: Zachary Amerman
# DATE CREATED: April 23, 2019
# REVISED DATE: April 24, 2019
# PURPOSE: Saves a model for later use and training

# Import torch libraries
import torch

def save_checkpoint(model, model_arch, model_hidden, epochs, criterion, optimizer, save_dir, class_to_idx, lr):
    model.class_to_idx = class_to_idx
    checkpoint = {'state_dict': model.state_dict(),
                  'epochs': epochs,
                  'crit_state': criterion.state_dict(),
                  'optim_state': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'arch': model_arch,
                  'hidden_units': model_hidden,
                  'lr': lr}
    
    # Save the new object
    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    return None