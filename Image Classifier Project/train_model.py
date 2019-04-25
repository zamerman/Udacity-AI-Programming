#!/usr/bin/env python
# PROGRAMMER: Zachary Amerman
# DATE CREATED: April 23, 2019
# REVISED DATE:
# PURPOSE: Trains a network with a user provided optimizer, criterion, # of epochs, training loader,
#       validation loader, and device

# Import torch libraries
import torch

def train_model(model, optimizer, criterion, epochs, trainloaders, validloaders, device):
    """
    Trains a network with an optimizer according to a criterion over a number of epochs
    on data provided by trainloaders and validloaders using device.
    Returns
    None
    """
    # Push model to cpu or cuda depending on device
    model = model.to(device)
    
    # Prepare containers for losses
    train_losses = []
    valid_losses = []
    
    for e in range(epochs):
        # Set model to training
        model.train()

        # Prepare a variable for the training loss
        train_loss = 0

        for images, labels in trainloaders:
            # Push images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # Zero the optimizer's gradient
            optimizer.zero_grad()

            # Feedforward the images
            log_ps = model.forward(images)

            # Calculate the loss
            loss = criterion(log_ps, labels)

            # Backpropagate through the gradient
            loss.backward()

            # Optimize
            optimizer.step()

            # Keep track of the running_loss
            train_loss += loss.item()

        else:
            # Prepare variables for validation loss and accuracy
            valid_loss = 0
            accuracy = 0

            # Turn off the gradient for this step as we aren't training and set model to evaluation
            with torch.no_grad():
                model.eval()

                for images, labels in validloaders:
                    # Push images and labels to device
                    images = images.to(device)
                    labels = labels.to(device)

                    # Feed the images forward through the model
                    log_ps = model.forward(images)

                    # Check loss and place in validation loss variable
                    valid_loss += criterion(log_ps, labels).item()

                    # Exponentiate to find the softmax
                    ps = torch.exp(log_ps)

                    # Select the top class
                    top_p, top_class = ps.topk(1, dim=1)

                    # Container for equality between model classification and actual labels
                    equals = top_class == labels.view(*top_class.shape)

                    # Calculate the accuracy
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Append training and validation losses to their containers
            train_losses.append(train_loss/len(trainloaders))
            valid_losses.append(valid_loss/len(validloaders))

            # Print loss results and accuracy
            print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(valid_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy/len(validloaders)))
    return None