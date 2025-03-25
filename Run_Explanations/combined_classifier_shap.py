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
    

class CombinedClassifierLogReg (nn.Module):    
    def __init__ (self, num_classes, axial_model, sagittal_model, coronal_model, device, dropout = 0.4, num_outputs = 512):
        super().__init__()

        self.device = device

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
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs),
        )
        self.sagittal_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs),
        )
        self.coronal_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs),
        )
   
        self.linear = torch.nn.Linear(num_outputs*3, num_classes)

    # make predictions
    def forward(self, x1, x2, x3):

        #x = torch.permute(x,(0, 3, 1, 2))

        #x = torch.split(x, [100,100,100], dim = 2)
        #x = x.reshape(3, len(x), 3, 100, 100)

        #x1 = x[:,:,000:100,:]#.contiguous()
        #x2 = x[:,:,100:200,:]#.contiguous()
        #x3 = x[:,:,200:300,:]#.contiguous()

        #x1 = x1
        #x2 = x2
        #x3 = x3

        #x1 = torch.permute(x1,(0, 3, 1, 2))
        #x2 = torch.permute(x2,(0, 3, 1, 2))
        #x3 = torch.permute(x3,(0, 3, 1, 2))

        #x1 = x1.to(self.device)
        #x2 = x2.to(self.device)
        #x3 = x3.to(self.device)

        y_axial = self.axial_model(x1)
        y_sagittal = self.sagittal_model(x2)
        y_coronal = self.coronal_model(x3)

        y_combined = torch.cat((y_axial, y_sagittal, y_coronal), dim = -1)

        output = torch.sigmoid(self.linear(y_combined))
        return output
    
class DoubleCombinedClassifierLogReg(torch.nn.Module):    

    def __init__ (self, num_classes, axial_model, sagittal_model, coronal_model, axial_model2, sagittal_model2, coronal_model2, device, dropout = 0.4, num_outputs = 512):
        super().__init__()

        self.num_outputs = num_outputs
        self.device = device

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

        self.axial_model2 = axial_model2
        for param in self.axial_model2.parameters():
            param.requires_grad_(False)

        self.sagittal_model2 = sagittal_model2
        for param in self.sagittal_model2.parameters():
            param.requires_grad_(False)

        self.coronal_model2 = coronal_model2
        for param in self.coronal_model2.parameters():
            param.requires_grad_(False)
        
        #Replacing classiferis in pretrained models.
        self.axial_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs*3),
        )
        self.sagittal_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs*3),
        )
        self.coronal_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs*3),
        )

        self.axial_model2.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs*2),
        )
        self.sagittal_model2.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs*2),
        )
        self.coronal_model2.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs*2),
        )
        """
        self.classifier = nn.Sequential(
        nn.SiLU(),
        nn.BatchNorm1d(num_features=num_outputs*6),    
        nn.Linear(num_outputs*6, 512*2),
        nn.SiLU(),
        nn.BatchNorm1d(num_features=512*2),
        nn.Linear(512*2, 128),
        nn.SiLU(),
        nn.BatchNorm1d(num_features=128),
        nn.Dropout(dropout),
        nn.Linear(128, num_classes),
        )"
        """

        self.linear = torch.nn.Linear(num_outputs * 15, num_classes)

    def forward(self, x1, x2, x3, x4, x5, x6):

        #x1, x2, x3, x4, x5, x6 = torch.unbind(x, dim=1)

        #print(x1[0].shape)
        #print(self.axial_model(x1).shape)
        """
        x1_out = torch.empty((len(x1), self.num_outputs*3))
        x2_out = torch.empty((len(x2), self.num_outputs*3))
        x3_out = torch.empty((len(x3), self.num_outputs*3))
        x4_out = torch.empty((len(x4), self.num_outputs*2))
        x5_out = torch.empty((len(x5), self.num_outputs*2))
        x6_out = torch.empty((len(x6), self.num_outputs*2))
        
        
        for i in range(0, len(x1)):

            x1_out[i] = self.axial_model(x1[i].unsqueeze(0)).squeeze()
            x2_out[i] = self.sagittal_model(x2[i].unsqueeze(0)).squeeze()
            x3_out[i] = self.coronal_model(x3[i].unsqueeze(0)).squeeze()
            x4_out[i] = self.axial_model2(x4[i].unsqueeze(0)).squeeze()
            x5_out[i] = self.sagittal_model2(x5[i].unsqueeze(0)).squeeze()
            x6_out[i] = self.coronal_model2(x6[i].unsqueeze(0)).squeeze()
        

        y_combined = torch.cat((x1_out, x2_out, x3_out, x4_out, x5_out, x6_out), dim = -1)
        
        """

        x1 = self.axial_model(x1)
        x2 = self.sagittal_model(x2)
        x3 = self.coronal_model(x3)
        x4 = self.axial_model2(x4)
        x5 = self.sagittal_model2(x5)
        x6 = self.coronal_model2(x6)

        y_combined = torch.cat((x1, x2, x3, x4, x5, x6), dim = -1)
        
        output = torch.sigmoid(self.linear(y_combined))

        return output
    
class DoubleCombinedClassifierLogRegTest(torch.nn.Module):    

    def __init__ (self, num_classes, axial_model, sagittal_model, coronal_model, axial_model2, sagittal_model2, coronal_model2, dropout = 0.4, num_outputs = 512):
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

        self.axial_model2 = axial_model2
        for param in self.axial_model2.parameters():
            param.requires_grad_(False)

        self.sagittal_model2 = sagittal_model2
        for param in self.sagittal_model2.parameters():
            param.requires_grad_(False)

        self.coronal_model2 = coronal_model2
        for param in self.coronal_model2.parameters():
            param.requires_grad_(False)
        
        #Replacing classiferis in pretrained models.
        self.axial_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs*3),
        )
        self.sagittal_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs*3),
        )
        self.coronal_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs*3),
        )

        self.axial_model2.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs*2),
        )
        self.sagittal_model2.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs*2),
        )
        self.coronal_model2.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_outputs*2),
        )
        """
        self.classifier = nn.Sequential(
        nn.SiLU(),
        nn.BatchNorm1d(num_features=num_outputs*6),    
        nn.Linear(num_outputs*6, 512*2),
        nn.SiLU(),
        nn.BatchNorm1d(num_features=512*2),
        nn.Linear(512*2, 128),
        nn.SiLU(),
        nn.BatchNorm1d(num_features=128),
        nn.Dropout(dropout),
        nn.Linear(128, num_classes),
        )"
        """

        self.linear = torch.nn.Linear(num_outputs * 15, num_classes)

    def forward(self, x1, x2, x3, x4, x5, x6, x7):

        if x7 == 'y_out':
            x1 = self.axial_model(x1)
            x2 = self.sagittal_model(x2)
            x3 = self.coronal_model(x3)
            x4 = self.axial_model2(x4)
            x5 = self.sagittal_model2(x5)
            x6 = self.coronal_model2(x6)

            output = [x1, x2, x3, x4, x5, x6, x7]

        elif x7 == 'normal':
            x1 = self.axial_model(x1)
            x2 = self.sagittal_model(x2)
            x3 = self.coronal_model(x3)
            x4 = self.axial_model2(x4)
            x5 = self.sagittal_model2(x5)
            x6 = self.coronal_model2(x6)

            y_combined = torch.cat((x1, x2, x3, x4, x5, x6), dim = -1)

            output = torch.sigmoid(self.linear(y_combined))

        else:
            y_combined = torch.cat((x1, x2, x3, x4, x5, x6), dim = -1)
            output = torch.sigmoid(self.linear(y_combined))



        return output