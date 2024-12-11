import ants
import dicom2nifti
import numpy as np
from Data_Preprocessing.mri_biascorrect import convert_and_correct
from mri_scale import scale_image
from mri_register import register_image
from mri_crop import crop_image

def preprocess(in_path, out_path, image_name):

    corrected_image = convert_and_correct(in_path)

    #Reads the brain atlas as an ANTS image.

    atlas_image = ants.image_read('//Users/olath/Downloads/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii')

    #Scales the image, stolen from Lucas

    scaled_image = scale_image(corrected_image, atlas_image)

    #Executes registration

    registered_image = register_image(scaled_image, atlas_image)

    #Crops the image to 100x100
    
    cropped_image = crop_image(registered_image)

    #Writes the preprocessedf image as a NIFTI image.

    ants.image_write(cropped_image,out_path + image_name + '.nii.gz')