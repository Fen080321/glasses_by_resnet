import torch
from torch import nn
from torchsummary import summary

class residual(nn.Module):
    def __init__(self, input_cahnnels  ,  num_cahnnels,use_1con = False,strides = 1 ):
        super(residual,self).__init__()
        self.relu = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels= input_cahnnels,out_channels=num_cahnnels,kernel_size=3,padding=1,stride=strides)
        self.c2 = nn.Conv2d(in_channels=num_cahnnels,out_channels=num_cahnnels,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(num_cahnnels)
        self.bn2 = nn.BatchNorm2d(num_cahnnels)
        if use_1con:
            self.c3 = nn.Conv2d(in_channels=input_cahnnels,out_channels=num_cahnnels,kernel_size=1,stride=strides)
        else:
            self.c3 = None

    def forward(self,x):
        y=self.relu(self.bn1(self.c1(x)))
        y=self.bn2(self.c2(y))
        if self.c3:
            x = self.c3(x)

        y = self.relu(x+y)
        return y
        

    

class resnet(nn.Module):
    def __init__(self,residual):
        super(resnet,self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.b2 = nn.Sequential(
            residual(64,64,use_1con=False,strides=1),
            residual(64,64,use_1con=False,strides=1),
        )
        self.b3 = nn.Sequential(
            residual(64,128,use_1con=True,strides=2),
            residual(128,128,use_1con=False,strides=1),
        )
        self.b4 = nn.Sequential(
            residual(128,256,use_1con=True,strides=2),
            residual(256,256,use_1con=False,strides=1),
        )
        self.b5 = nn.Sequential(
            residual(256,512,use_1con=True,strides=2),
            residual(512,512,use_1con=False,strides=1),
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512,26)
        )
        

    def forward(self,x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x

if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mymodel=resnet(residual).to(device)
    print(summary(mymodel,(3,224,224)))
