import os
import pandas as pd
import numpy as np
import torchvision
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from augmentation import augment
import math

device = torch.device('mps' if torch.mps.is_available() else 'cuda')

class MRI_Dataset(Dataset):

    def __init__(self, annotations_file, transform=None, target_transform=None):

        self.reference_table = pd.read_csv(annotations_file)
        self.img_dir = '/Users/olath/Documents/ADNI_SLICED/'
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.reference_table)

    def __getitem__(self, idx):

        'Generates one sample of data'
          
        # Select sample
        sample = self.reference_table.iloc[idx]
        image_path = (self.img_dir + str(sample['IMAGE_ID']) +'/'+ sample['ORIENTATION'] +'/'+ str(sample['SLICE']) + '.png')
        y = sample['CLASS']

        # Load data and get label
        X = read_image(image_path)

        X = X.to(device)

        if self.transform is not None:
           X = self.transform(X)
            
        X = X.repeat(3, 1, 1)
        X = X/X.max()
        
                         
        return X, y