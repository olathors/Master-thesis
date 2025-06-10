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

    def __init__(self, annotations_file, img_dir, transform=None, slice = None):

        self.reference_table = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.slice = slice

    def __len__(self):
        return len(self.reference_table)

    def __getitem__(self, idx):

        'Generates one sample of data'
          
        # Select sample
        sample = self.reference_table.iloc[idx]

         
        image_path_axial_1 = (self.img_dir + str(sample['IMAGE_ID']) +'/AXIAL/'+ str(self.slice[0]) + '.png')
        image_path_sagittal_1 = (self.img_dir + str(sample['IMAGE_ID']) +'/SAGITTAL/'+ str(self.slice[1]) + '.png')
        image_path_coronal_1 = (self.img_dir + str(sample['IMAGE_ID']) +'/CORONAL/'+ str(self.slice[2]) + '.png')

        image_path_axial_2 = (self.img_dir + str(sample['IMAGE_ID']) +'/AXIAL/'+ str(self.slice[3]) + '.png')
        image_path_sagittal_2 = (self.img_dir + str(sample['IMAGE_ID']) +'/SAGITTAL/'+ str(self.slice[4]) + '.png')
        image_path_coronal_2 = (self.img_dir + str(sample['IMAGE_ID']) +'/CORONAL/'+ str(self.slice[5]) + '.png')

        image_path_axial_3 = (self.img_dir + str(sample['IMAGE_ID']) +'/AXIAL/'+ str(self.slice[6]) + '.png')
        image_path_sagittal_3 = (self.img_dir + str(sample['IMAGE_ID']) +'/SAGITTAL/'+ str(self.slice[7]) + '.png')
        image_path_coronal_3 = (self.img_dir + str(sample['IMAGE_ID']) +'/CORONAL/'+ str(self.slice[8]) + '.png')

        image_path_axial_4 = (self.img_dir + str(sample['IMAGE_ID']) +'/AXIAL/'+ str(self.slice[9]) + '.png')
        image_path_sagittal_4 = (self.img_dir + str(sample['IMAGE_ID']) +'/SAGITTAL/'+ str(self.slice[10]) + '.png')
        image_path_coronal_4 = (self.img_dir + str(sample['IMAGE_ID']) +'/CORONAL/'+ str(self.slice[11]) + '.png')
        

        y = sample['CLASS']

        # Load data and get label
        X1 = read_image(image_path_axial_1)
        X2 = read_image(image_path_sagittal_1)
        X3 = read_image(image_path_coronal_1)
        X4 = read_image(image_path_axial_2)
        X5 = read_image(image_path_sagittal_2)
        X6 = read_image(image_path_coronal_2)
        X7 = read_image(image_path_axial_3)
        X8 = read_image(image_path_sagittal_3)
        X9 = read_image(image_path_coronal_3)
        X10 = read_image(image_path_axial_4)
        X11 = read_image(image_path_sagittal_4)
        X12 = read_image(image_path_coronal_4)
             


        if self.transform is not None:
            X1 = self.transform(X1) 
            X2 = self.transform(X2) 
            X3 = self.transform(X3) 
            X4 = self.transform(X4) 
            X5 = self.transform(X5) 
            X6 = self.transform(X6) 
            X7 = self.transform(X7) 
            X8 = self.transform(X8) 
            X9 = self.transform(X9) 
            X10 = self.transform(X10) 
            X11 = self.transform(X11) 
            X12 = self.transform(X12) 

        X1 = X1.repeat(3, 1, 1)
        X2 = X2.repeat(3, 1, 1)
        X3 = X3.repeat(3, 1, 1)
        X4 = X4.repeat(3, 1, 1)
        X5 = X5.repeat(3, 1, 1)
        X6 = X6.repeat(3, 1, 1)
        X7 = X7.repeat(3, 1, 1)
        X8 = X8.repeat(3, 1, 1)
        X9 = X9.repeat(3, 1, 1)
        X10 = X10.repeat(3, 1, 1)
        X11 = X11.repeat(3, 1, 1)
        X12 = X12.repeat(3, 1, 1)
            
            

        X1 = X1/X1.max()
        X2 = X2/X2.max()
        X3 = X3/X3.max()
        X4 = X4/X4.max()
        X5 = X5/X5.max()
        X6 = X6/X6.max()
        X7 = X7/X7.max()
        X8 = X8/X8.max()
        X9 = X9/X9.max()
        X10 = X10/X10.max()
        X11 = X11/X11.max()
        X12 = X12/X12.max()

                              
        return torch.stack((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12), dim= 0), y