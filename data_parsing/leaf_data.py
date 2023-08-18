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

import json

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

images_path = {
    "celeba": "/ws/data_parsing/leaf/data/celeba/data/raw/img_align_celeba"
}

labels_path = {
    "celeba": "/ws/data_parsing/leaf/data/celeba/data/raw/list_attr_celeba.txt"
}

def process_img(img_path, mode):
    img = Image.open(img_path).convert('RGB')
    transformed_img = data_transforms[mode](img)
    return transformed_img

def read_celeba_labels(labels_path, target_label="Smiling"):
    with open(labels_path) as lfile:
        data = lfile.readlines()[1:]
    headers = data[0].strip().split()
    rows = data[1:]
    
    index = 0
    for i in range(len(headers)):
        if headers[i] == target_label:
            index = i
            break

    labels_map = {}
    for r in rows:
        spl = r.strip().split()
        f = spl[0]

        labels_map[f] = spl[index + 1] == 1
        #labels_map[f] = {p[0]: p[1] for p in zip(headers, f[1:])}
    return labels_map

class LEAFDataset():
    def __init__(self, dname):
        self.image_path = images_path[dname]

        self.dname = dname
        self.input_dim = 3072

        # Celeba binary classification
        # FEMNIST 47 classes

        if dname == "celeba":
            self.labels_table = read_celeba_labels(labels_path['celeba'])

        #self.classes = self.dataset_orig['train'].classes
        #self.dataset_sizes = {x: len(self.dataset_orig[x]) for x in phases}
        self.dataset = {}

    def convert_img(self, device, feature_extractor=None, BS=128):
        files = os.listdir(self.image_path)
        self.features_map = {}
        for i in tqdm(range(0, len(files), BS)):
            slice = files[i:i+BS]
            images = [process_img(os.path.join(self.image_path, fname), mode='train') for fname in slice]
            inputs = torch.stack(images, dim=0)
            #labels = [self.labels_table[f] for f in slice]
            if feature_extractor != None:
                x = apply_feature_extractor(inputs, feature_extractor, device)
                x = torch.squeeze(x)
                x_numpy = x.numpy()

                self.features_map.update({f: (torch.from_numpy(x_numpy[i]), self.labels_table[f]) for i, f in enumerate(slice)})

    def save_img_features(self, file_path):
        with open(file_path, "wb") as pickle_f:
            dump = self.features_map
            pickle.dump(dump, pickle_f)

    def load_features(self, file_path):
        with open(file_path, "rb") as pickle_f:
            self.features_map = pickle.load(pickle_f)

    ra
if __name__=="__main__":
    f_extractor = ResDenseConcat()

    GPU = 1
    device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')

    f_extractor.eval()
    f_extractor.to(device)

    dname = "celeba"

    datapath = {
        "celeba": "/ws/data_parsing/leaf/data/celeba/data",
        "femnist": ""
    }

    dpath = datapath[dname]
    dset = LEAFDataset(dname)
    dset.convert_img(device, f_extractor)
    dset.save_img_features("../out/celebaresdensefeatures.pkl")
