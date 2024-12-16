import ants
import dicom2nifti
import numpy as np
from mri_biascorrect import convert_and_correct
from mri_scale import scale_image
from mri_register import register_image
from mri_crop import crop_image

def preprocess_image(in_path, out_path, image_name, atlas_image):

    #Converts input series to NIFTI, then ANTS image and preforms bias field correction.

    corrected_image = convert_and_correct(in_path, image_name)

    #Scales the input image according to the atlas, mostly stolen from Lucas

    scaled_image = scale_image(corrected_image, atlas_image)

    #Executes registration of the input image to the atlas image.

    registered_image = register_image(scaled_image, atlas_image)

    #Crops the input image to 100x100x100.
    
    cropped_image = crop_image(registered_image)

    #Writes the preprocessed input image as a NIFTI image.

    ants.image_write(cropped_image,out_path + image_name + '.nii.gz')