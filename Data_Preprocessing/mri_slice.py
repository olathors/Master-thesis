import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from mri_crop import crop_image

def slice_image(path, image):
    
    img_nifti = nib.load(path + image)

    img = np.array(img_nifti.dataobj)
    """
    if img.shape != (100, 100, 100):

        center_dim = [int(np.ceil(x/2)) for x in img.shape]
        lower_dim = [int(x - 50) for x in center_dim]
        upper_dim = [int(x + 50) for x in center_dim]

        img = img[lower_dim[0]:upper_dim[0], lower_dim[1]:upper_dim[1], lower_dim[2]:upper_dim[2]]

    """
    
    saggital_slices = [0] * 100
    coronal_slices = [0] * 100
    axial_slices = [0] * 100

    for i in range(0,100):

        saggital_slices[i] = generate_slice_sagittal(i, img)   
        coronal_slices[i] = generate_slice_coronal(i, img)
        axial_slices[i] = generate_slice_axial(i, img)

    return [saggital_slices, coronal_slices, axial_slices]

def generate_slice_sagittal(slice_index, img):

    rot = np.rot90(img, k=3, axes=(1,2)).copy()
    rot = np.rot90(rot, k=2, axes=(0,2)).copy()
    return  rot[slice_index,:,:]
            

def generate_slice_coronal(slice_index, img):

    rot = np.rot90(img, k=3, axes=(0,2)).copy()
    return  rot[:,slice_index,:]

def generate_slice_axial(slice_index, img):

    rot = np.rot90(img, k=3, axes=(0,1)).copy()        
    return  rot[:,:,slice_index]
