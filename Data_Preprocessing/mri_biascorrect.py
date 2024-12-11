import dicom2nifti
import ants
    
def convert_and_correct(in_path):

    dicom2nifti.dicom_series_to_nifti(in_path, 'temp.nii.gz', reorient_nifti=True)

    #Reads the NIFTI image as an ANTS image.

    ants_image = ants.image_read('temp.nii.gz')

    #Executes bias field correction.

    corrected_image = ants.n4_bias_field_correction(ants_image)

    #Returns ants image
    
    return corrected_image