import dicom2nifti
import ants
    
def convert_and_correct(in_path, image_name):

    '''
    Function that converts dcm series to nifti and then ants image, and executes bias field correction.

    --PARAMETERS--

    in_path: Path to the zipped folder containing the dcm series for a single image

    image_name: name of the relevant image

    returns: bias corrected ants image
    
    '''

    try:
        dicom2nifti.dicom_series_to_nifti(in_path, 'temp.nii.gz', reorient_nifti=True)
    except Exception as err:
        logf = open("conversion.log", "a")
        logf.write(("Failed to convert {0}: {1}\n".format(image_name, str(err))))
        logf.close()

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