import torch
from torchvision import datasets, models, transforms
import numpy as np

class ResNet101Wrapper(torch.nn.Module):
    def __init__(self):
        super(ResNet101Wrapper, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.newResNet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        for param in self.newResNet.parameters():
            param.requires_grad = False
    def forward(self,x):
        x = self.newResNet.forward(x)
        return x

class DenseNet121Wrapper(torch.nn.Module):
    def __init__(self):
        super(DenseNet121Wrapper, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        for param in self.densenet.parameters():
            param.requires_grad = False
        self.newDenseNet = torch.nn.Sequential(*(list(self.densenet.children())[:-1]))
        for param in self.newDenseNet.parameters():
            param.requires_grad = False
            
    def forward(self,x):
        x = self.newDenseNet.forward(x)
        return x 

class ResDenseConcat:
    def __init__(self):        
        super(ResDenseConcat, self).__init__()
        self.resnet = ResNet101Wrapper() #models.resnet101(pretrained=True)
        self.densenet = DenseNet121Wrapper() #models.densenet121(pretrained=True)

    def __call__(self, x):
        y = x.detach().clone()
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        #x = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
        #print(x.shape)

        y = self.densenet(y)
        y = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))(y)
        y = torch.flatten(y, 1)
        #print(y.shape)

        x = torch.cat((x, y), 1) 
        #x = np.squeeze(x, axis=(2,3))
        #print(x.shape)
        return x

    def eval(self):
        self.resnet.eval()
        self.densenet.eval()
        return

    def to(self, device):
        self.resnet.to(device)
        self.densenet.to(device)
        return

class Res101:
    # Wrapper
    def __init__(self):        
        super(Res101, self).__init__()
        self.resnet = ResNet101Wrapper()

    def __call__(self, x):
        x = self.resnet(x)

        return x

    def eval(self):
        self.resnet.eval()
        return

    def to(self, device):
        self.resnet.to(device)
        return

