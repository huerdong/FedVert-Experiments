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

phases = ['train', 'test']

transforms = {
        'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                    [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            ]),
        'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                    [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            ])
    }

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

class CIFAR10Dataset():
    def __init__(self, path):
        self.dataset_orig = {x: datasets.CIFAR10(path, train=(x=='train'), download=True, transform=transforms[x]) for x in phases}
        self.classes = self.dataset_orig['train'].classes
        self.dataset_sizes = {x: len(self.dataset_orig[x]) for x in phases}
        self.dataset = {}

    def convert_features(self, file_path, device, feature_extractor=None, BS=128):
        if feature_extractor is None:
            self.dataset['train'] = self.dataset_orig['train']
            self.dataset['test'] = self.dataset_orig['test']
            return
        for phase in ['train', 'test']:
            loader = DataLoader(self.dataset_orig[phase], batch_size=BS, shuffle=False)
            self.dataset[phase] = ExtractedFeatureDataset("cifar10", [], self.dataset_orig[phase])
            for (inputs, labels) in tqdm(loader):
                x = apply_feature_extractor(inputs, feature_extractor, device)
                x_numpy = x.numpy()
                self.dataset[phase].dataset.extend(((torch.from_numpy(x_numpy[i]), l) for i, l in enumerate(labels)))

        with open(file_path, "wb") as pickle_f:
            dump = {phase: self.dataset[phase].dataset for phase in ['train', 'test']}
            pickle.dump(dump, pickle_f)

    def load_features(self, file_path):
        with open(file_path, "rb") as pickle_f:
            load_pkl = pickle.load(pickle_f)
            self.dataset['train'] = ExtractedFeatureDataset("cifar10", load_pkl['train'], self.dataset_orig['train'])
            self.dataset['test'] = ExtractedFeatureDataset("cifar10", load_pkl['test'], self.dataset_orig['test'])

if __name__=="__main__":
    dset = CIFAR10Dataset("./out/data/cifar10")
    f_extractor = ResDenseConcat()

    GPU = 1
    device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')

    f_extractor.eval()
    f_extractor.to(device)
    dset.convert_features("./out/cifar10resdensefeatures.pkl", device, f_extractor)