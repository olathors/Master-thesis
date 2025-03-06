import torch
from torch import nn

class CombinedClassifierL (nn.Module):
    def __init__ (self, num_classes, axial_model, sagittal_model, coronal_model, dropout = 0.4, num_outputs = 512):
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
            nn.Linear(1280, num_outputs),
        )
        self.sagittal_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_outputs),
        )
        self.coronal_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_outputs),
        )

        #New common classifier.
        self.classifier = nn.Sequential(
        nn.SiLU(),
        nn.BatchNorm1d(num_features=num_outputs*3),    
        nn.Linear(num_outputs*3, 512),
        nn.SiLU(),
        nn.BatchNorm1d(num_features=512),
        nn.Linear(512, 128),
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
    
class CombinedClassifierLogReg(torch.nn.Module):    

    def __init__ (self, num_classes, axial_model, sagittal_model, coronal_model, dropout = 0.4, num_outputs = 512):
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
    def forward(self, x):

        x1, x2, x3 = torch.unbind(x, dim=1)
        y_axial = self.axial_model(x1)
        y_sagittal = self.sagittal_model(x2)
        y_coronal = self.coronal_model(x3)
        y_combined = torch.cat((y_axial, y_sagittal, y_coronal), dim = -1)

        output = torch.sigmoid(self.linear(y_combined))
        return output