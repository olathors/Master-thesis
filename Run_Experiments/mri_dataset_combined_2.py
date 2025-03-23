import os
import pandas as pd
import numpy as np
import torchvision
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
#from augmentation import augment
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

class MRI_Dataset(Dataset):

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, slice = None, orientation = None, triple = False):

        self.reference_table = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.triple = triple
        self.slice = slice
        """
        if slice is not None:
            self.reference_table['SLICE'] = slice

        if orientation is not None:
            self.reference_table['ORIENTATION'] = orientation
        """


    def __len__(self):
        return len(self.reference_table)

    def __getitem__(self, idx):

        'Generates one sample of data'
          
        # Select sample
        sample = self.reference_table.iloc[idx]

         
        image_path_axial = (self.img_dir + str(sample['IMAGE_ID']) +'/AXIAL/'+ str(self.slice[0]) + '.png')
        image_path_sagittal = (self.img_dir + str(sample['IMAGE_ID']) +'/SAGITTAL/'+ str(self.slice[1]) + '.png')
        image_path_coronal = (self.img_dir + str(sample['IMAGE_ID']) +'/CORONAL/'+ str(self.slice[2]) + '.png')

        image_path_axial_2 = (self.img_dir + str(sample['IMAGE_ID']) +'/AXIAL/'+ str(self.slice[3]) + '.png')
        image_path_sagittal_2 = (self.img_dir + str(sample['IMAGE_ID']) +'/SAGITTAL/'+ str(self.slice[4]) + '.png')
        image_path_coronal_2 = (self.img_dir + str(sample['IMAGE_ID']) +'/CORONAL/'+ str(self.slice[5]) + '.png')
        

        y = sample['CLASS']

        # Load data and get label
        X1 = read_image(image_path_axial)
        X2 = read_image(image_path_sagittal)
        X3 = read_image(image_path_coronal)
        X4 = read_image(image_path_axial_2)
        X5 = read_image(image_path_sagittal_2)
        X6 = read_image(image_path_coronal_2)
             


        if self.transform is not None:
            X1 = self.transform(X1) 
            X2 = self.transform(X2) 
            X3 = self.transform(X3) 
            X4 = self.transform(X4) 
            X5 = self.transform(X5) 
            X6 = self.transform(X6) 

        X1 = X1.repeat(3, 1, 1)
        X2 = X2.repeat(3, 1, 1)
        X3 = X3.repeat(3, 1, 1)
        X4 = X4.repeat(3, 1, 1)
        X5 = X5.repeat(3, 1, 1)
        X6 = X6.repeat(3, 1, 1)
            
            

        X1 = X1/X1.max()
        X2 = X2/X2.max()
        X3 = X3/X3.max()
        X4 = X4/X4.max()
        X5 = X5/X5.max()
        X6 = X6/X6.max()

                              
        return torch.stack((X1, X2, X3, X4, X5, X6), dim= 0), y