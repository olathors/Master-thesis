import os
from mri_slice import slice_image 
from PIL import Image
from matplotlib import pyplot as plt

def main(in_path, out_path):

    for subdir, dirs, files in os.walk(in_path):
        for file in files:
            slices = slice_image(in_path, file)
            image_name = file[:-7]

            os.makedirs(out_path+image_name+'/saggital')
            os.makedirs(out_path+image_name+'/coronal')
            os.makedirs(out_path+image_name+'/axial')

            for i in range(0, 100):

                arr = slices[0][i]
                im = Image.fromarray(arr)
                im = im.convert('L')
                plt.imsave(out_path+image_name+'/saggital/'+str(i)+'.png', im, cmap='gray')

                arr = slices[1][i]
                im = Image.fromarray(arr)
                im = im.convert('L')
                plt.imsave(out_path+image_name+'/coronal/'+str(i)+'.png', im, cmap='gray')
                
                arr = slices[2][i]
                im = Image.fromarray(arr)
                im = im.convert('L')
                plt.imsave(out_path+image_name+'/axial/'+str(i)+'.png', im, )
                
main('/Volumes/Extreme SSD/ADNI_PROCESSED/', '/Volumes/Extreme SSD/ADNI_SLICED/')