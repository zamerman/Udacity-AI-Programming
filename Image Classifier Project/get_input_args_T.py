#!/usr/bin/env python
# PROGRAMMER: Zachary Amerman
# DATE CREATED: April 23, 2019
# REVISED DATE:
# PURPOSE: Creates a function that retrieves command line inputs from the user.
#       Command Line Arguments:
#       1. Image Directory as data_dir
#       2. Save Directory as --save_dir with default value 'checkpoints'
#       3. Network Architecture as --arch with default value 'vgg11'
#       4. Switch for GPU calculations as --gpu, default value is 'cpu' and when placed in the line changes it to 'cuda'
#       5. Learning rate as --learning_rate with default value 0.01
#       6. Hidden units as --hidden_units with default value 512
#       7. Epochs as --epochs with default value 5

# Import argparse library
import argparse

def get_input_args():
    """
    Enables command line arguments in order to control directories, network architectures,
    choose your device, and control training variables.
    Parameters
     None
    Returns
     parser.parse_args() - a data structure that stores the command line arguments object
    """
    # Setup command line interface and retrieve arguments
    parser = argparse.ArgumentParser(description='''Create and train a new network on a dataset and save the model as a checkpoint''')
    parser.add_argument('data_dir', type=str, help='The path to the training and validation datasets')
    parser.add_argument('--save_dir', dest='save_dir', default='checkpoints', help='The directory where checkpoints are stored')
    parser.add_argument('--arch', dest='arch', default='vgg11', help='''The pretrained architecture to use. Possible values are "alexnet", "vgg11", "squeezenet1_0", and "densenet121"''')
    parser.add_argument('--learning_rate', type=float, dest='lr', default=0.01,
                        help='Determines model learning rate')
    parser.add_argument('--hidden_units', type=int, dest='hidden_units', default=512,
                        help='Determines number of hidden units in hidden layer')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5,
                        help='Determines number of epochs over which the model is trained')
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda', default='cpu',
                        help='Determines if architecture uses gpu or not')

    # Store command line arguments
    return parser.parse_args()