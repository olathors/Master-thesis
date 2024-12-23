import ants
import numpy as np

def scale_image (corrected_image, atlas_image):
    
    '''
    Function for clipping outliers and scaling intensity of image to atlas.

    --PARAMETERS--

    corrected_image: Image to be clipped/scaled

    atlas_image: Atlas image

    returns: Clipped and scaled ants image.
    
    '''

    image_array = corrected_image.numpy()

    lower_threshold, upper_threshold = get_percentiles(image_array)

    image_clipped = clip_image_intensity(image_array, lower_threshold, upper_threshold)

    lower_atlas_threshold, upper_atlas_threshold = get_atlas_thresholds(atlas_image)

    image_scaled = scale_image_linearly(image_clipped,lower_atlas_threshold,upper_atlas_threshold)

    moving_image = ants.from_numpy(image_scaled, direction=corrected_image.direction)

    return moving_image

def get_percentiles(img, lower = 0.02, upper = 99.8):

    flat_image = img.ravel()
    lower_percentile = np.percentile(flat_image,lower)
    upper_percentile = np.percentile(flat_image,upper)
    return lower_percentile, upper_percentile

def scale_image_linearly(image_array, lower_threshold, upper_threshold):

    scaled_image = (image_array - lower_threshold) / (upper_threshold - lower_threshold)
    return scaled_image

def clip_image_intensity(image,lower_threshold,upper_threshold):

    image[image > upper_threshold] = upper_threshold
    image[image < lower_threshold] = lower_threshold
    return image

def get_atlas_thresholds(atlas_image):

    atlas_array = atlas_image.numpy()
    return get_percentiles(atlas_array)