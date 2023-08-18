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

from feature_extractors import *

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

class BirdsDataset(Dataset):
    def __init__(self, root_dir, mode, num_classes=200, image_size=(256, 256)):
        # Birds dataset has by default 200 classes and 256x256 images
        super(BirdsDataset, self).__init__()
        self.image_size = image_size
        self.mode = mode
        self.data = pd.read_csv(os.path.join(root_dir, "birds_%s.csv" % mode))
        self.drop_index = self.data[self.data['class_id'] < num_classes].index
        self.data = self.data.iloc[self.drop_index]
        self.images_dir = os.path.join(root_dir, "images")
        classes_path = os.path.join(root_dir, "classes.txt")
        with open(classes_path) as class_file:
            pairs = [r.split() for r in class_file.readlines()]
            self.classes = {p[0]: p[1] for p in pairs}

        self.targets = self.data.iloc[:,-1:].to_numpy().flatten() # labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item, crop=True):
        # Actually I don't know if I can manually call the crop to be true?
        '''
        Get image and its label at index `idx`.
        It crops (If specified) the image according to the bounding
        box indicated in the csv file.
        '''
        img_path = os.path.join(self.images_dir,self.data.iloc[item]['path'])
        bbox = self.data.iloc[item][['x', 'y', 'width', 'height']]
        img = Image.open(img_path).convert('RGB')  
        #img.save("toriginal.jpg")      
        if crop:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[0]+bbox[2]
            y2 = bbox[1]+bbox[3]
            img = img.crop([x1, y1, x2, y2])
        back_to_img_transform = transforms.ToPILImage()
        transform = {
            'train': transforms.Compose([
                # Do we want these random transformations
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(10),
                #transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                #transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        }

        # Is cropping needed?
        # if crop:
        #     x1 = bbox[0]
        #     y1 = bbox[1]
        #     x2 = bbox[0]+bbox[2]
        #     y2 = bbox[1]+bbox[3]
        #     img = img.crop([x1, y1, x2, y2])
        #     transform = torchvision.transforms.Compose([
        #         # resize it to the size indicated by `image_size`
        #         torchvision.transforms.Resize(self.image_size),
        #         # convert it to a tensor
        #         torchvision.transforms.ToTensor(),
        #         # normalize it to the range [−1, 1]
        #         torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        #     ])

        x = transform[self.mode](img)
        back_to_img = back_to_img_transform(x)
        #back_to_img.save("test.jpg")
        d = self.data.iloc[item]['class_id']
        #print(d)
        #input()
        return x, d

    def number_of_classes(self):
        return self.data['class_id'].max() + 1


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

class CUBBirdsDataset():
    def __init__(self, path):
        self.input_dim = 2000
        self.dataset_orig = {x: BirdsDataset(path, x) for x in phases}
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
    dset = CUBBirdsDataset("./CUB_200_2011")
    #f_extractor = Res101()
    f_extractor = ResDenseConcat()

    GPU = 1
    device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')

    f_extractor.to(device)
    dset.convert_features("./out/cubbirdsresdensefeatures.pkl", device, f_extractor)
