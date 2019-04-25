# Import torch library
import torch

def evaluate_image(image_tensor, model, topk, device):
    # Push the model and image to whichever device was selected
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Set model to evaluation mode and evaluate
    model.eval()
    log_ps = model(image_tensor)
    
    # Exponentiate the result the retrieve the soft max probability values
    ps = torch.exp(log_ps)
    
    # Grab the topk values and classes
    top_p, top_class = ps.topk(topk)
    
    # Convert the tensors to lists
    probs = top_p.tolist()[0]
    classes = top_class.tolist()[0]
    
    return probs, classes