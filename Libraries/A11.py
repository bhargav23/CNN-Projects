from tqdm import tqdm_notebook, tnrange
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.07

class newResnet(nn.Module):
    def __init__(self):
        super(newResnet, self).__init__()

        self.PrepLayer = nn.Sequential(
          nn.Conv2d(3, 64, 3,padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.Dropout(dropout_value)      
        )

        self.layer1convblock1 = nn.Sequential(
          nn.Conv2d(64, 128, 3,padding=1),
          nn.MaxPool2d(2, 2),
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.Dropout(dropout_value)      
        )

        self.resnet1 = nn.Sequential(
          nn.Conv2d(128, 128, 3,padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Conv2d(128, 128, 3,padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Dropout(dropout_value) 
        )

        self.layer2convblock2 = nn.Sequential(
            nn.Conv2d(128, 256, 3,padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value)      
        )
       
        self.layer3convblock3 = nn.Sequential(
          nn.Conv2d(256, 512, 3,padding=1),
          nn.MaxPool2d(2, 2),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.Dropout(dropout_value)      
        )

        self.resnet3 = nn.Sequential(
          nn.Conv2d(512, 512, 3,padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Conv2d(512, 512, 3,padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Dropout(dropout_value) 
        )
        
        self.pool1 = nn.MaxPool2d(4, 1)


        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) 


    def forward(self, x):
      
        x = self.PrepLayer(x)

        x = self.layer1convblock1(x)
        r1 = self.resnet1(x)
        x = torch.add(x, r1)

        x = self.layer2convblock2(x)

        x = self.layer3convblock3(x)
        r3 = self.resnet3(x)
        x = torch.add(x, r3)

        x = self.pool1(x) 
  
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)