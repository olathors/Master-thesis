import os
from mri_slice import slice_image 
from PIL import Image
from matplotlib import pyplot as plt
import torch
import numpy as np
import shutil

def main(in_path, out_path):

    for subdir, dirs, files in os.walk(in_path):
        for file in files:
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

            shutil.make_archive(out_path+image_name, 'zip', 'temp_slicing_folder')
            shutil.rmtree('temp_slicing_folder')
                
main('/Volumes/Extreme SSD/ADNI_PROCESSED/', '/Volumes/Extreme SSD/ADNI_SLICED/')