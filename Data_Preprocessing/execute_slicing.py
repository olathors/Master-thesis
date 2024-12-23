import os
from mri_slice import slice_image 
from PIL import Image
import numpy as np
import shutil
import time
from mri_preprocessed_fetch import get_preprocessed_imagenames

def main(in_path, out_path, global_time):

    images_already_preprocessed = get_preprocessed_imagenames(out_path)

    for subdir, dirs, files in os.walk(in_path):

        total_images_sliced = 0
        percentage = 0

        for file in files:

            if (file[:-7]+'.zip') in images_already_preprocessed:
                print('Image already sliced!')
                files.remove(file)
            
            elif (not file.startswith('._')):

                total_images_to_slice = len(files)
                slices = slice_image(in_path, file)
                image_name = file[:-7]        

                os.makedirs('temp_slicing_folder'+'/saggital')
                os.makedirs('temp_slicing_folder'+'/coronal')
                os.makedirs('temp_slicing_folder'+'/axial')

                for i in range(0, 100):

                    arr = slices[0][i]
                    arr *= 255.0/arr.max()
                    im = Image.fromarray(arr)
                    im = im.convert('L')
                    im.save('temp_slicing_folder'+'/saggital/'+str(i)+'.png')

                    arr = slices[1][i]
                    arr *= 255.0/arr.max()
                    im = Image.fromarray(arr)
                    im = im.convert('L')
                    im.save('temp_slicing_folder'+'/coronal/'+str(i)+'.png')
                    
                    arr = slices[2][i]
                    arr *= 255.0/arr.max()
                    im = Image.fromarray(arr)
                    im = im.convert('L')
                    im.save('temp_slicing_folder'+'/axial/'+str(i)+'.png')

                total_images_sliced += 1
                prev_percentage = percentage
                percentage = round(100 * float(total_images_sliced)/float(total_images_to_slice))

                if prev_percentage != percentage:
                    time_spent = (time.time() - global_time)/60
                    time_remaining = (time_spent/percentage) * (100 - percentage)
                    print('Slicing images is %.0f' % percentage,'%','complete and will finish in %.2f' % time_remaining, 'minutes. ')

                shutil.make_archive(out_path+image_name, 'zip', 'temp_slicing_folder')
                shutil.rmtree('temp_slicing_folder')

global_time = time.time()

main('/Volumes/Extreme SSD/ATLAS/', '/Volumes/Extreme SSD/ATLAS_SLICED/', global_time)