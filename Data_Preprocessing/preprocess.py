import ants
import dicom2nifti
import numpy as np

def preprocess(in_path, out_path, image_name):

    #Converts the dicom file series into a single NIFTI image.

    dicom2nifti.dicom_series_to_nifti(in_path, 'temp.nii.gz', reorient_nifti=True)

    #Reads the NIFTI image as an ANTS image.

    ants_image = ants.image_read('temp.nii.gz')

    #Executes bias field correction.

    moving_image = ants.n4_bias_field_correction(ants_image)

    #Reads the brain atlas as an ANTS image.

    fixed_image = ants.image_read('//Users/olath/Downloads/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii')

    #Executes registration

    mytx = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyN' ,grad_step=0.1)

    mywarpedimage = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=mytx['fwdtransforms'])

    #Crops the image to 100x100
    
    center_dim = [int(np.ceil(x/2)) for x in mywarpedimage.shape]
    lower_dim = [int(x - 50) for x in center_dim]
    upper_dim = [int(x + 50) for x in center_dim]

    cropped_image =  ants.crop_indices(mywarpedimage,lowerind = lower_dim,upperind = upper_dim)

    #Writes the preprocessedf image as a NIFTI image.

    ants.image_write(cropped_image,out_path + image_name + '.nii.gz')