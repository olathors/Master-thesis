import dicom2nifti
import ants
    
def convert_and_correct(in_path):

    dicom2nifti.dicom_series_to_nifti(in_path, 'temp.nii.gz', reorient_nifti=True)

    #Reads the NIFTI image as an ANTS image.

    ants_image = ants.image_read('temp.nii.gz')

    #If image is a time series, takes only one image.

    if len(ants_image.shape) == 4:

        image_array = ants_image.numpy()

        fixed_image_array = image_array[:, :, :, 0]

        ants_image = ants.from_numpy(fixed_image_array)

    #Executes bias field correction.

    corrected_image = ants.n4_bias_field_correction(ants_image)

    #Returns ants image
    
    return corrected_image