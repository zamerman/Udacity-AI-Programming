#!/usr/bin/env python
# PROGRAMMER: Zachary Amerman
# DATE CREATED: April 23, 2019
# REVISED DATE:
# PURPOSE: Creates a function that retrieves command line inputs from the user.
#       Command Line Arguments:
#       1. Image Path as /path/to/image
#       2. Checkpoint as checkpoint
#       3. Top K most likely classes as --top_k, default value is 3
#       4. Category mapping with --category_names references a json file, default value is cat_to_name.json
#       5. Switch for GPU calculations as --gpu, default value is 'cpu' and when placed in the line changes it to 'cuda'

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
    parser = argparse.ArgumentParser(description='''Predicts the flower name shown in an image along with the probability of that name''')
    parser.add_argument('image', metavar='/path/to/image', type=str, help='The path to the image')
    parser.add_argument('checkpoint', type=str, help='The checkpoint containing the model')
    parser.add_argument('--category_names', type=str, dest='cat_names', default='cat_to_name.json',
                        help='The json file containing the category mapping')
    parser.add_argument('--top_k', type=int, dest='top_k', default=1,
                        help='Determines how many of the most probable classes are shown')
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda', default='cpu',
                        help='Determines if architecture uses gpu or not')

    return parser.parse_args()