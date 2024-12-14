from mri_preprocess import preprocess_image
import ants
import numpy as np
import time
import zipfile
import shutil

def main(in_data, images):

    data = np.load(images,allow_pickle='TRUE').item()

    atlas_image = ants.image_read('//Users/olath/Downloads/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii')
    counter = 0
    images_left = len(data)
    global_time = time.time()

    archive = zipfile.ZipFile(in_data)

    for key in data:
        local_time = time.time()

        for file in archive.namelist():
            if file.endswith('I'+key+'.dcm'):
                archive.extract(file, 'temp_folder')


        preprocess_image('temp_folder', '/Users/olath/Documents/GitHub/Master-thesis/Data_Preprocessing/test_data/', 'I'+str(key), atlas_image)
        counter += 1
        images_left -= 1
        local_time_spent = (time.time() - local_time)
        global_time_spent = (time.time() - global_time) / 60
        print ('Processed image nr:',counter,'- id:',key, 'in %.2f' % local_time_spent ,"seconds.")
        print('There are', images_left, 'images left to preprocess.')
        print('Total time spent is %.2f' % global_time_spent, "minutes.")
        print('\n')

        shutil.rmtree('temp_folder')



main('/Volumes/Extreme SSD/Download/Download_collection1.zip', "/Users/olath/Documents/GitHub/Master-thesis/Data_Experimentation/images_with_paths_file_1.npy")