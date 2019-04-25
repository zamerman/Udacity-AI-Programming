#!/usr/bin/env python
# PROGRAMMER: Zachary Amerman
# DATE CREATED: April 23, 2019
# REVISED DATE:
# PURPOSE: Predicts the name of a flower shown in an image along with the probability of that name

# Import torch library
import torch

# Import json library
import json

# Import custom functions
from get_input_args_P import get_input_args
from load_checkpoint import load_checkpoint
from process_image import process_image
from evaluate_image import evaluate_image

def main():
    # Get command line arguments
    args = get_input_args()
    
    # Grab json name to class mapping
    with open(args.cat_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # Load model and adjust device
    model, epochs, criterion, optimizer = load_checkpoint(args.checkpoint)
        
    # Load and process image to numpy array
    image_array = process_image(args.image)
    
    # Convert the numpy array to a torch tensor of the float tensor type and unsqueeze it in order to place evaluate it
    image_tensor = torch.from_numpy(image_array).type(torch.FloatTensor)
    image_tensor = image_tensor.unsqueeze_(0) 
    
    # Evaluate images
    probs, classes = evaluate_image(image_tensor, model, args.top_k, args.device)
    # Convert the classes values into labels
    labels = []
    for label in classes:
        for key in model.class_to_idx:
            if model.class_to_idx[key] == label:
                labels.append(key)
    
    # Convert labels to flower names
    names = [cat_to_name[label] for label in labels]
    
    # Print out results
    print("Flower : Probability")
    for i in range(len(names)):
        print("%s : %f.3" % (names[i], probs[i]))

if __name__ == "__main__":
    main()