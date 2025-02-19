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

        if slice is not None:
            self.reference_table['SLICE'] = slice

        if orientation is not None:
            self.reference_table['ORIENTATION'] = orientation


    def __len__(self):
        return len(self.reference_table)

    def __getitem__(self, idx):

        'Generates one sample of data'
          
        # Select sample
        sample = self.reference_table.iloc[idx]

         
        image_path = (self.img_dir + str(sample['IMAGE_ID']) +'/'+ str(sample['ORIENTATION']).lower() +'/'+ str(sample['SLICE']) + '.png')
        

        y = sample['CLASS']

        # Load data and get label
        X = read_image(image_path)


        if self.triple:
            image_path_1 = (self.img_dir + str(sample['IMAGE_ID']) +'/'+ str(sample['ORIENTATION']).lower() +'/'+ str(sample['SLICE']+2) + '.png')
            image_path_3 = (self.img_dir + str(sample['IMAGE_ID']) +'/'+ str(sample['ORIENTATION']).lower() +'/'+ str(sample['SLICE']-2) + '.png')
            X3 = read_image(image_path_3)
            X1 = read_image(image_path_1)
            X = torch.cat((X1, X, X3), dim=0)

            if self.transform is not None:
                X = self.transform(X) 
             
        else:
            X = X.repeat(3, 1, 1)
            
            if self.transform is not None:
                X = self.transform(X) 

        X = X/X.max()
                              
        return X, y