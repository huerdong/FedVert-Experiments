import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import pickle
from feature_extractors import *
import pandas as pd
from skimage import io
from PIL import Image

phases = ['train', 'test']

# transforms = {
#         'train': transforms.Compose([
#             transforms.Resize((256,256)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(10),
#             transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
#             transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ]),
#         'test': transforms.Compose([
#             transforms.Resize((256,256)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#     }

def apply_feature_extractor(tensor, feature_extractor, device):
    if feature_extractor is None:
        return tensor
    gpu_tensor = tensor.to(device)
    cpu_tensor = feature_extractor(gpu_tensor).detach().cpu()
    return cpu_tensor

class ExtractedFeatureDataset(Dataset):
    def __init__(self, name, data, dataset_orig):
        self.name = name
        self.dataset = data
        self.classes = dataset_orig.classes
        self.targets = dataset_orig.targets

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class CUBBirdsDataset():
    def __init__(self, path):
        self.input_dim = 3072
        self.dataset_orig = {x: datasets.ImageFolder(os.path.join(path, x), data_transforms[x])
                  for x in phases}

        #self.dataset_orig = {x: datasets.ImageFolder(path + f"/{x.capitalize}", transform=transforms[x]) for x in phases}
        self.classes = self.dataset_orig['train'].classes
        self.dataset_sizes = {x: len(self.dataset_orig[x]) for x in phases}
        self.dataset = {}

    def convert_features(self, file_path, device, feature_extractor=None, BS=32):
        if feature_extractor is None:
            self.dataset['train'] = self.dataset_orig['train']
            self.dataset['test'] = self.dataset_orig['test']
            return
        for phase in ['train', 'test']:
            loader = DataLoader(self.dataset_orig[phase], batch_size=BS, shuffle=False)
            self.dataset[phase] = ExtractedFeatureDataset("cubbirds", [], self.dataset_orig[phase])
            for (inputs, labels) in tqdm(loader):
                x = apply_feature_extractor(inputs, feature_extractor, device)
                x = torch.squeeze(x)
                #print(x.shape)
                x_numpy = x.numpy()
                self.dataset[phase].dataset.extend(((torch.from_numpy(x_numpy[i]), l) for i, l in enumerate(labels)))

        with open(file_path, "wb") as pickle_f:
            dump = {phase: self.dataset[phase].dataset for phase in ['train', 'test']}
            pickle.dump(dump, pickle_f)

    def load_features(self, file_path):
        with open(file_path, "rb") as pickle_f:
            load_pkl = pickle.load(pickle_f)
            self.dataset['train'] = ExtractedFeatureDataset("cubbirds", load_pkl['train'], self.dataset_orig['train'])
            self.dataset['test'] = ExtractedFeatureDataset("cubbirds", load_pkl['test'], self.dataset_orig['test'])

if __name__=="__main__":
    dset = CUBBirdsDataset("./CUB_200_2011/images_new")
    #f_extractor = Res101()
    f_extractor = ResDenseConcat()

    GPU = 1
    device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')

    f_extractor.eval()
    f_extractor.to(device)
    #dset.convert_features("../out/cubbirdsres101features.pkl", device, f_extractor)
    dset.convert_features("../out/cubbirdsresdensefeatures.pkl", device, f_extractor)
