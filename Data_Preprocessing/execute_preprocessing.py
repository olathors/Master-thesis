from mri_preprocess import preprocess_image
from mri_preprocessed_fetch import get_preprocessed_imagenames
import ants
import numpy as np
import time
import zipfile
import shutil
import os

def main(in_data, images, out_path, global_time, process_no):

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

    for key in relevant_image_names:

        current_image_no += 1
        images_left_counter -= 1

        if (str(key)+'.nii.gz') in images_already_preprocessed:
            print('Image',current_image_no,'already preprocessed')

        else:
            local_time = time.time()

            try:
                os.makedirs('temp_folder')

                for file in raw_image_archive.namelist():
                    
                    #if file.endswith('I'+key+'.dcm'):
                    if ('I'+key) in file:
                        raw_image_archive.extract(file, 'temp_folder')

                preprocess_image('temp_folder', out_path, str(key), atlas_image)

            except Exception as err:

                logf = open("conversion.log", "a")
                logf.write(("Failed to convert {0}: {1}\n".format(str(key), str(err))))
                logf.close()

            

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
out_path = '/Volumes/Extreme SSD/ADNI_PROCESSED/'

global_time = time.time()

main(in_path, image_list_path, out_path, global_time, 10)