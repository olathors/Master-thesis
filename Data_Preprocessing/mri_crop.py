import ants
import numpy as np


def crop_image(registered_image):

    center_dim = [int(np.ceil(x/2)) for x in registered_image.shape]
    lower_dim = [int(x - 50) for x in center_dim]
    upper_dim = [int(x + 50) for x in center_dim]

    cropped_image =  ants.crop_indices(registered_image,lowerind = lower_dim,upperind = upper_dim)

    return cropped_image