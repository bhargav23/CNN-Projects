from tqdm import tqdm_notebook, tnrange
import torch.nn as nn
import torch.nn.functional as F
import torch

dropout_value = 0.07
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
          nn.Conv2d(1, 32, 3,padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(dropout_value)
        )

        self.convblock2 = nn.Sequential(
          nn.Conv2d(32, 32, 3,padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(dropout_value)
        )

        self.convblock3 = nn.Sequential(
          nn.Conv2d(32, 32, 3,padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(dropout_value)      
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock4 = nn.Sequential(
	      nn.Conv2d(32, 64, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(dropout_value)
        )

        self.convblock5 = nn.Sequential(
        nn.Conv2d(64, 64, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(dropout_value)
        )

     

        self.pool2 = nn.MaxPool2d(2, 2)

        self.dconvblock1 = nn.Sequential(
	      nn.Conv2d(64, 128, 3,dilation=2,padding=2),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)
        )

        self.convblock6 = nn.Sequential(
        nn.Conv2d(64, 128, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)
        )

        self.convblock7 = nn.Sequential(
        nn.Conv2d(128, 128, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)      
        )
        '''
        self.pool3 = nn.MaxPool2d(2, 2)

        self.convblock8 = nn.Sequential(
        nn.Conv2d(64, 128, 1,groups=64,dilation=1,padding=1),
        nn.Conv2d(128, 256, kernel_size=(1,1))
        
        
          
        )

        self.convblock9 = nn.Sequential(
        nn.Conv2d(256, 256, 1,groups=256,dilation=1,padding=1),
        nn.Conv2d(256, 256, kernel_size=(1,1)),
           
        )

        '''
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4,4))
        ) # output_size = 1

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) 




    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool2(x)
        x1 = self.dconvblock1(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = torch.add(x, x1)

        #x = self.pool3(x)
        #x = self.convblock8(x)
        #x = self.convblock9(x)

        x = self.gap(x)        
        x = self.convblock10(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class NETS6(nn.Module):
    def __init__(self):
        super(NETS6, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
