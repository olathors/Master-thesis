import ants
import numpy as np


def crop_image(registered_image):

    '''
    Function for center cropping image to 100x100x100.

    --PARAMETERS--
    
    registered_image: Image to be cropped
    
    returns: Cropped image
    
    '''

    center_dim = [int(np.ceil(x/2)) for x in registered_image.shape]
    lower_dim = [int(x - 50) for x in center_dim]
    upper_dim = [int(x + 50) for x in center_dim]


    #169×208×179

    cropped_image =  ants.crop_indices(registered_image,lowerind = lower_dim,upperind = upper_dim)

    return cropped_image