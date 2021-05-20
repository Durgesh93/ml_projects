import torch
import torch.nn as nn
import config




class OrcaDectNet(nn.Module):

    def __init__(self): 
        super(OrcaDectNet, self).__init__()


        self.conv2d_1     = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=3, padding=2)
        self.activation_1 = nn.ReLU()
        self.dropout_1    = nn.Dropout()
        self.conv2d_2     = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=3, padding=2)
        self.activation_2 = nn.ReLU()
        self.dropout_2    = nn.Dropout(p=0.6)
        self.conv2d_3     = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=3, padding=2)
        self.activation_3 = nn.ReLU()
        self.dropout_3    = nn.Dropout()
        self.conv2d_4     = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=3, padding=2)
        self.activation_4 = nn.ReLU()
        self.dropout_4    = nn.Dropout(p=0.6)
        self.flatten_1    = nn.Flatten()
        self.linear_1     = nn.Linear(in_features=1536, out_features=256)
        self.activation_5 = nn.ReLU()
        self.linear_2     = nn.Linear(in_features=256, out_features=2)
        


    def forward(self,x):
        #print(x.shape)
        out = self.conv2d_1(x)
        #print(out.shape)    
        out = self.activation_1(out)
        #print(out.shape)
        out = self.dropout_1(out)
        #print(out.shape)   
        out = self.conv2d_2(out)
        #print(out.shape)    
        out = self.activation_2(out)
        #print(out.shape)
        out = self.dropout_2(out)
        #print(out.shape)   
        out = self.conv2d_3(out)
        #print(out.shape)    
        out = self.activation_3(out)
        #print(out.shape)
        out = self.dropout_3(out)
        #print(out.shape)   
        out = self.conv2d_4(out)
        #print(out.shape)    
        out = self.activation_4(out)
        #rint(out.shape)
        out = self.dropout_4(out)
        #print(out.shape)
        out = self.flatten_1(out)
        #print(out.shape)
        out = self.linear_1(out)
        #print(out.shape)
        out = self.activation_5(out)
        #print(out.shape)
        out = self.linear_2(out)
        #print(out.shape)
        return out
        
