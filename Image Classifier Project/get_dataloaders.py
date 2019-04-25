#!/usr/bin/env python
# PROGRAMMER: Zachary Amerman
# DATE CREATED: April 23, 2019
# REVISED DATE:
# PURPOSE: A function to grab our dataloaders for training and validation.

# Import torch and torchvision libraries
import torch
from torchvision import datasets, transforms

def get_dataloaders(data_dir):
    """
    Grabs dataloaders for training and validation from the data directory specified by the user.
    Parameters:
     data_dir - The directory which contains the training and validation data
    Returns:
     dataloaders - A dictionary with two keys containing our trainand and validation dataloaders
    """
    # Define directory paths for training and validation
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # Create normalize transform and then create transforms for training and validation
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(256),
                                                    transforms.RandomRotation(45),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    normalize]),
                       'valid': transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    normalize])
                      }

    # Load datasets with directory paths and transforms
    image_datasets = {'train': datasets.ImageFolder(train_dir, data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, data_transforms['valid'])}

    # Define dataloaders
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True)}
    
    return dataloaders, image_datasets['train'].class_to_idx