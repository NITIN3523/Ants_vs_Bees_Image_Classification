import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3) # (60 + 2 *0 -3 / 1) + 1 = 58*58*16
        self.pool1 = nn.MaxPool2d(2) # (58 + 2 *0 -2 / 2) + 1 = 29*29*16
        
        self.conv2 = nn.Conv2d(16, 32, 3) # (29 + 2 *0 -3 / 1) + 1 = 27*27*32
        self.pool2 = nn.MaxPool2d(2) # (27 + 2 *0 -2 / 2) + 1 = 13*13*32
        
        self.conv3 = nn.Conv2d(32, 64, 3) # (13 + 2 *0 -3 / 1) + 1 = 11*11*64
        self.pool3 = nn.MaxPool2d(2)   # (11 + 2 *0 -2 / 2) + 1 = 5*5*64
        
        self.flatten = nn.Flatten() # 5*5*64 = 1600
        
        self.fc1 = nn.Linear(64*5*5,256)    
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,2)
        
        self.softmax = nn.Softmax(dim=1) # Ne1 = 0.2, Ne2 = 0.8 ==> 1
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
         
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x
