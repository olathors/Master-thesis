import torch
from torch import nn
import numpy as np
import sys

class CombinedClassifierL (nn.Module):
    def __init__ (self, num_classes, axial_model, sagittal_model, coronal_model, dropout = 0.4):
        super().__init__()

        #Freezing pretrained models.
        self.axial_model = axial_model
        for param in self.axial_model.parameters():
            param.requires_grad_(False)

        self.sagittal_model = sagittal_model
        for param in self.sagittal_model.parameters():
            param.requires_grad_(False)

        self.coronal_model = coronal_model
        for param in self.coronal_model.parameters():
            param.requires_grad_(False)

        #Replacing classiferis in pretrained models.
        self.axial_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 512),
        )
        self.sagittal_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 512),
        )
        self.coronal_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 512),
        )

        #New common classifier.
        self.classifier = nn.Sequential(
        nn.SiLU(),
        nn.BatchNorm1d(num_features=512*3),    
        nn.Linear(512*3, 512),
        nn.SiLU(),
        nn.BatchNorm1d(num_features=512),
        nn.Linear(512, 128),
        nn.SiLU(),
        nn.BatchNorm1d(num_features=128),
        nn.Dropout(dropout),
        nn.Linear(128, num_classes),
        )
    
    def forward(self, x):

        #if not torch.is_tensor(x):
        #    x = torch.from_numpy(x)

        #x = x.to(self.device)

        x = torch.permute(x,(0, 3, 1, 2))

        x = torch.split(x, [100,100,100], dim = 2)

        x1 = x[0].contiguous()
        x2 = x[1].contiguous()
        x3 = x[2].contiguous()

        y_axial = self.axial_model(x1)
        y_sagittal = self.sagittal_model(x2)
        y_coronal = self.coronal_model(x3)
        y_combined = torch.cat((y_axial, y_sagittal, y_coronal), dim = -1)
        output = self.classifier(y_combined)
        return  output
    
class CombinedClassifierM (nn.Module):
    def __init__ (self, num_classes, axial_model, sagittal_model, coronal_model, dropout = 0.4):
        super().__init__()

        #Freezing pretrained models.
        self.axial_model = axial_model
        for param in self.axial_model.parameters():
            param.requires_grad_(False)

        self.sagittal_model = sagittal_model
        for param in self.sagittal_model.parameters():
            param.requires_grad_(False)

        self.coronal_model = coronal_model
        for param in self.coronal_model.parameters():
            param.requires_grad_(False)

        #Replacing classiferis in pretrained models.
        self.axial_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, 512),
        )
        self.sagittal_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, 512),
        )
        self.coronal_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, 512),
        )

        #New common classifier.
        self.classifier = nn.Sequential(
        nn.SiLU(),
        nn.BatchNorm1d(num_features=512*3),
        nn.Linear(512*3, 128),
        nn.SiLU(),
        nn.BatchNorm1d(num_features=128),
        nn.Dropout(dropout),
        nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x1, x2, x3 = torch.unbind(x, dim=1)
        y_axial = self.axial_model(x1)
        y_sagittal = self.sagittal_model(x2)
        y_coronal = self.coronal_model(x3)
        y_combined = torch.cat((y_axial, y_sagittal, y_coronal), dim = -1)
        output = self.classifier(y_combined)
        return  output