import torch
from torchvision import datasets, models, transforms

class ResDenseConcat:
    def __init__(self):        
        super(ResDenseConcat, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.densenet = models.densenet121(pretrained=True)

    def __call__(self, x):
        x = torch.unsqueeze(x, dim=0) # Convert to a batch
        y = x.detach().clone()
        x = self.resnet(x)
        y = self.densenet(y)

        x = torch.cat((x, y), 1) 
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
        self.resnet = models.resnet101(pretrained=True)

    def __call__(self, x):
        x = torch.unsqueeze(x, dim=0) # Convert to a batch
        x = self.resnet(x)

        return x

    def eval(self):
        self.resnet.eval()
        return

    def to(self, device):
        self.resnet.to(device)
        return
