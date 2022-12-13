from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from collections import OrderedDict


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, params_DoubleConv, init = False):
        super(DoubleConv, self).__init__()
        
        ordered_dict = OrderedDict([("conv1",nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=params_DoubleConv["bias"]))])
        
        if params_DoubleConv["BatchNorm"]:
            ordered_dict["BatchNorm1"] = nn.BatchNorm2d(out_channels)
        ordered_dict["activation1"] = params_DoubleConv["activation"]
        if params_DoubleConv["use_dropout"] != None:
            ordered_dict["Dropout1"] = nn.Dropout(p=params_DoubleConv["p_dropout"])
            
        ordered_dict["conv2"] = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=params_DoubleConv["bias"])
        if params_DoubleConv["BatchNorm"]:
            ordered_dict["BatchNorm2"] = nn.BatchNorm2d(out_channels)
        ordered_dict["activation2"] = params_DoubleConv["activation"]
        if params_DoubleConv["p_dropout"] != None:
            ordered_dict["Dropout2"] = nn.Dropout(p=params_DoubleConv["p_dropout"])
            
        if init:
            nn.init.xavier_normal_(ordered_dict["conv1"].weight)
            nn.init.xavier_normal_(ordered_dict["conv2"].weight)
            
            if params_DoubleConv["bias"]:
                ordered_dict["conv1"].bias.data.fill_(0.01)
                ordered_dict["conv2"].bias.data.fill_(0.01)
        
        self.conv = nn.Sequential(ordered_dict)
        

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, params_DoubleConv, params_up ,in_channels=3, out_channels=1, features=[64, 128, 256, 512],init = False,
    ): # outchannels = 1: binary class
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature,params_DoubleConv,init))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            ordered_dict = OrderedDict([])
            if params_up["BatchNorm"]:
                ordered_dict["BatchNorm"] = nn.BatchNorm2d(feature*2)
                
            ordered_dict["trans_conv"] = nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2,)
            
            if init:    
                nn.init.xavier_normal_(ordered_dict["trans_conv"].weight)
                if params_DoubleConv["bias"]:
                    ordered_dict["trans_conv"].bias.data.fill_(0.01)
                    
            if params_up["use_dropout"]:
                ordered_dict["Dropout"] = nn.Dropout(p=params_up["p_dropout"])
                
            self.ups.append(nn.Sequential(ordered_dict))
            
            self.ups.append(DoubleConv(feature*2, feature,params_DoubleConv,init))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2,params_DoubleConv,init)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        if init:
            nn.init.xavier_normal_(self.final_conv.weight)
            if params_DoubleConv["bias"]:
                self.final_conv.bias.data.fill_(0.01)
            

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:]) # height and width

            concat_skip = torch.cat((skip_connection, x), dim=1) # concatenate along in-channel axis
            
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class UNET_no_skip_connection(nn.Module):
    def __init__(
            self,params_DoubleConv, in_channels=3, out_channels=1, features=[64, 128, 256, 512],init=False
    ): # outchannels = 1: binary class
        super(UNET_no_skip_connection, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature,params_DoubleConv,init))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                ) # feature*2: skip connection
            )
            
            if init:    
                nn.init.xavier_normal_(self.ups[-1].weight)
                if params_DoubleConv["bias"]:
                    self.ups[-1].bias.data.fill_(0.01)
            
            self.ups.append(DoubleConv(feature, feature,params_DoubleConv,init))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2,params_DoubleConv,init)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        if init:
            nn.init.xavier_normal_(self.final_conv.weight)
            if params_DoubleConv["bias"]:
                self.final_conv.bias.data.fill_(0.01)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:]) # height and width
                
            x = self.ups[idx+1](x)

        return self.final_conv(x)




