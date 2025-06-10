from mri_preprocess import preprocess_image
from mri_preprocessed_fetch import get_preprocessed_imagenames
import ants
import numpy as np
import time
import zipfile
import shutil
import os
import sys

def main(in_data, images, out_path, global_time, process_no):

    '''
    Main function for running the preprocessing pipeline.

    --PARAMETERS--

    in_data: Path to the zipped download file from ADNI

    images: Path to dictionary containing image names and specific paths

    out_path: Path to output directory where preprocessed images are saved.

    global_time: To keep track of time over multiple function calls.

    process_no: To keep track of call no.

    '''

    #Serring up variables.
    relevant_image_names = np.load(images,allow_pickle='TRUE').item()
    atlas_image = ants.image_read('/Volumes/Extreme SSD/Download/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii')
    images_preprocessed_counter = 0
    running_average = 0
    current_image_no = 0
    images_left_counter = len(relevant_image_names)
    total_images = len(relevant_image_names)
    images_already_preprocessed = get_preprocessed_imagenames(out_path)
    raw_image_archive = zipfile.ZipFile(in_data)

    if os.path.exists('/Users/olath/Documents/GitHub/Master-thesis/Data_Preprocessing/temp_folder'):
        shutil.rmtree('temp_folder')

    #Iterating through all image names.
    for key in relevant_image_names:
        current_image_no += 1
        images_left_counter -= 1

        #Checking if image already preprocessed.
        if (str(key)+'.nii.gz') in images_already_preprocessed:
            print('Image',current_image_no,'already preprocessed')

        #If not already preprocessed, runs preprocessing.
        else:
            local_time = time.time()

            try:
                #Folder to store extracted dcm series.
                if os.path.exists('temp_folder'):
                    shutil.rmtree('temp_folder')
                os.makedirs('temp_folder')

                for file in raw_image_archive.namelist():                 
                    if ('I'+key) in file:                       
                        raw_image_archive.extract(file, 'temp_folder')

                #Executes the preprocessing
                preprocess_image('temp_folder', out_path, str(key), atlas_image, save_steps= True)

            except Exception as err:
                logf = open("conversion.log", "a")
                logf.write(("Failed to convert {0}: {1}\n".format(str(key), str(err))))
                logf.close()
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
    
            #Metrics
            images_preprocessed_counter += 1
            local_time_spent = (time.time() - local_time)
            global_time_spent = (time.time() - global_time) / 60
            running_average = global_time_spent / images_preprocessed_counter
            percentage = 100 * float(current_image_no)/float(total_images)

            print('\n')
            print('Preprocessing batch', process_no, 'out of 10 is %.0f' % percentage,'%','complete.')
            print ('Processed image nr:',current_image_no,'- id:',key, 'in %.2f' % local_time_spent ,"seconds.")
            print('There are', images_left_counter, 'images left of this batch to preprocess.')
            print('Total time spent is %.2f' % global_time_spent, "minutes of batch no:", process_no)
            print('At this speed, this batch will finish in %.2f' % (running_average * images_left_counter) ,'minutes.')

            if os.path.exists('/Users/olath/Documents/GitHub/Master-thesis/Data_Preprocessing/temp_folder'):
                shutil.rmtree('temp_folder')

in_path = '/Volumes/Extreme SSD/Download/Download_collection_dataset.zip'
image_list_path = "/Users/olath/Documents/GitHub/Master-thesis/Data_Experimentation/images_paths/images_with_paths_file_9.npy"
out_path = '/Users/olath/Documents/Preprocessing_steps_noscale/'

global_time = time.time()

main(in_path, image_list_path, out_path, global_time, 10)