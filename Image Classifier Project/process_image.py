from PIL import Image
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open up the image using pil
    im = Image.open(image)
    
    # Resize the image
    im = im.resize((256, 256))
    
    # Center crop the image
    w, h = im.size
    n_w, n_h = (224, 224)
    l = (w - n_w)/2
    t = (h - n_h)/2
    r = (w + n_w)/2
    b = (h + n_h)/2
    im = im.crop((l, t, r, b))
    
    # Convert the image to array
    np_image = np.array(im)
    
    # Change values to floats between 0-1
    np_image = np_image/255
    
    # Normalize array
    np_image = (np_image - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    
    # Transpose the array for pytorch
    np_image = np_image.transpose(2, 0, 1)
    return np_image