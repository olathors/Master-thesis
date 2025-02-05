import os
import pandas as pd
import numpy as np
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

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

        # Load data and get label
        X = read_image(image_path)
        X = X.repeat(3, 1, 1)
        X = X/X.max()
        

        #if 'ROTATION_ANGLE' in sample.index and sample['ROTATION_ANGLE'] != 0:
        #    X = ndimage.rotate(X, sample['ROTATION_ANGLE'], reshape=False)
                         
        y = sample['CLASS']
        return X, y