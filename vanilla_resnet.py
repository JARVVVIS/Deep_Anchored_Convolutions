'''
Implementing vanilla DACNN with ResNet-18 as base
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 

class SmallBlock(nn.Module):
    def __init__(self,conv,ch):
        super(SmallBlock,self).__init__()
        self.conv = conv 
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self,x):
        output = F.relu(self.bn1(self.conv(x)))
        output = self.bn2(self.conv(x))
        output += x ## Residual addition
        output = F.relu(output)
        return output

class DACNN_RES(nn.Module):
    def __init__(self,classes=10,blocks=[3,4,6,3]):
        super(DACNN_RES,self).__init__()
        self.classes = classes
        self.blocks = blocks

        self.channels = 128 ## number of channels for fixed conv.

        ### Init Layer
        self.conv_1 = nn.Conv2d(3,128,3,padding=1,bias=False)
        self.bn_1   = nn.BatchNorm2d(self.channels)
        ## Global(Fixed) Convolution
        self.conv_global = nn.Conv2d(128,128,3,padding=1,bias=False)
        self.max_pool = nn.MaxPool2d(2,stride=2)
        sections = []
                
        for idx in range(4):
            sections.append(self.resnet_module(idx,128))
        self.sections = nn.Sequential(*sections)

        self.clf_in = 128
        self.clf = nn.Linear(self.clf_in,self.classes)

    def resnet_module(self,idx,channel):
        '''
        Return Basic ResNet Block => F(x)+x
        '''
        section = self.blocks[idx]
        layers = []
        for _ in range(section):
            layers.extend([SmallBlock(self.conv_global,channel),self.max_pool])
        return nn.Sequential(*layers)
    
    def forward(self,x):
        output = F.relu(self.bn_1(self.conv_1(x))) ## Init Layer
        output = self.sections(output).view(-1,self.clf_in)
        output = self.clf(output)
        return output


if __name__ == '__main__':
    model = DACNN_RES()
    x = torch.rand((64,3,64,64))
    ans = model(x)
    print(x.shape,ans.shape)