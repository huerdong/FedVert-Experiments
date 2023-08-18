import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from models.models import MLP, MLPTwoHidden
from data_parsing.cifar10_data import CIFAR10Dataset
from data_parsing.cifar100_data import CIFAR100Dataset
from data_parsing.cubbirds_data_version2 import CUBBirdsDataset
from data_parsing.dirichlet_sampler import *
from data_parsing.feature_extractors import *
from models.proximal_sgd import ProximalOptimizer

from utils.plot import MetricTracker
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--gpu", type=int, required=True)
args = parser.parse_args()

GPU = args.gpu # Change depending on others' use of GPUS
dname = args.dataset

EPOCHS=2000 # If this isn't improving by 50 epochs I don't think it will ever converge
#LOCAL_EPOCHS = [1]
LOCAL_EPOCHS=[25]
#LR = [0.01]
LR=[0.01]
MU=0.2 # I think usually this works best
BS=32
USERS= 100 # Number of users
#RAC={100: [0.1,0.3], 1000: [0.02, 0.06]} # Fraction of users to report
#FRAC=FRAC[USERS]
FRAC=[0.1]
#DIRIC= 0.01 # Dirichlet parameter 
DIRIC =[0.01]

# Early cutoff threshold and sliding window
# If no improvement more than early cutoff then terminate
early_cutoff = 0.005
#out_dir = "fedavg_full_model_acc"
out_dir = f"fedprox_fullmodel18_{dname}_res"
os.makedirs(f"out/{out_dir}", exist_ok=True)

status_dir = "status"
os.makedirs(f"out/{status_dir}", exist_ok=True)

feature_extractor = ResDenseConcat()
feature_extractor.eval()

modes = ['train', 'test']

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

if dname == "cifar100":
    dataset = {x: datasets.CIFAR100("./out/data/cifar100", train=(x=='train'), download=True, transform=data_transforms[x]) for x in modes}
    class_names = dataset['train'].classes
    dataset_sizes = {x: len(dataset[x]) for x in modes}
    OUTPUT_SIZE=100
elif dname == "aircraft":
    data_dir = os.path.join("data_parsing", "aircraft", "fgvc-aircraft-2013b", "data", "images_new")
    dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in modes}
    dataset_sizes = {x: len(dataset[x]) for x in modes}
    class_names = dataset['train'].classes
    OUTPUT_SIZE=100
elif dname == "dtextures":
    data_dir = os.path.join("data_parsing", "describe_textures", "dtd", "images_new")
    dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in modes}
    dataset_sizes = {x: len(dataset[x]) for x in modes}
    class_names = dataset['train'].classes
    OUTPUT_SIZE=47
elif dname == "stanfordcars":
    dataset = {x: datasets.StanfordCars("./out/data/stanfordcars", download=True, transform=data_transforms[x]) for x in modes}
    targets_f = "./out/stanfordcars_targets.pkl"
    import pickle
    with open(targets_f, "rb") as pickle_targets:
        dataset['train'].targets = pickle.load(pickle_targets)        
    class_names = dataset['train'].classes
    dataset_sizes = {x: len(dataset[x]) for x in modes}
    OUTPUT_SIZE=196
elif dname == "vggflowers":
    data_dir = os.path.join("data_parsing", "vggflowers", "images_new")
    dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in modes}
    dataset_sizes = {x: len(dataset[x]) for x in modes}
    class_names = dataset['train'].classes
    OUTPUT_SIZE=102
elif dname == "cubbirds":
    data_dir = os.path.join("data_parsing", "CUB_200_2011", "images_new")
    dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in modes}
    dataset_sizes = {x: len(dataset[x]) for x in modes}
    class_names = dataset['train'].classes
    OUTPUT_SIZE=200

input_dim = (BS, 3, 224, 224)

#classes = dataset.classes
#dataset_sizes = dataset.dataset_sizes

dataset['train'].name = dname

class DatasetSplit(Dataset):
    # Uses a partitioning of dataset indices according to users
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

loaders = {'test': DataLoader(dataset['test'], batch_size=BS, shuffle=True)}
#loaders['train'] = [DataLoader(DatasetSplit(dataset.dataset['train'], usersplit_train[user]), batch_size=BS, shuffle=True) for user in range(USERS)]

device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')

# Implements Fed average aggregation
# Modify this for different algorithms
def aggregate(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        if "conv" in k or "fc" in k:
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    del w
    return w_avg

def local_train():
    # TODO Move some of the training stuff here to make it modular 
    return

def get_params(net):
    param_group = [copy.deepcopy(t) for t in net.parameters()]
    return param_group

# Add scheduler?
def train_model_federated(num_users, part_ratio, device, model, local_model, feature_extractor, criterion, mu, diric, learning_rate, num_epochs=25, local_epochs=1):
    since = time.time()
    tracker = MetricTracker(since)

    #model.to(device)
    #local_model.to(device)
    #feature_extractor.to(device)
    #feature_extractor.eval()

    sum_model = copy.deepcopy(model)
    #glob_sum = summary(sum_model, input_dim, verbose=0)
    local_sum = summary(sum_model, input_dim, verbose=0)
    local_model_size = local_sum.total_param_bytes
    est_inputs = (torch.randn(input_dim),)
    flops = FlopCountAnalysis(model, est_inputs)
    flops_estimate = flops.total()
    del local_sum
    del sum_model
    del flops

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_test_acc = 0.0

    if USERS > 1:
        usersplit_train = dirichlet_sampler_idsize(dataset['train'], USERS, diric)
    else:
        # If one user use whole dataset on one node
        usersplit_train = {0: np.arange(len(dataset['train']))}

    # For FedProx/FedAvg just use global model for model trained locally
    local_model_copy = copy.deepcopy(model).to(device)
    optimizers = [optim.SGD(local_model_copy.parameters(), lr=learning_rate, momentum=0.0) for user in range(num_users)]
    schedulers = [lr_scheduler.StepLR(optimizers[user], step_size=num_epochs//4, gamma=0.1) for user in range(num_users)]

    timeout = False
    for epoch in range(num_epochs):
        local_lr = learning_rate
        if epoch == int(num_epochs / 2):
            local_lr = learning_rate/10
        if epoch == int(num_epochs * 0.75):
            local_lr = learning_rate/100

        model.train()
        w_glob = model.state_dict()

        print(f'Epoch {epoch}/{num_epochs -1}')
        print('-' * 10)
        
        local_weights = []
        local_loss = []
        
        phase = 'train' # For federated cases 'train' mode in user loop then 'test'

        # Implement user participation
        part_users= max(1, int(num_users * part_ratio)) # Minimum one user
        idxs_users = np.random.choice(range(num_users), part_users, replace=False)
        idxs_users = sorted(idxs_users) # Sort this so that printing makes sense

        for user in idxs_users:
            #torch.cuda.empty_cache()

            #print(f'User {user}/{num_users - 1}')

            local_params = copy.deepcopy(w_glob)
            local_model_copy.load_state_dict(local_params)

            local_model_copy.train()

            running_loss = 0.0
            running_corrects = 0
            user_total_trials = 0

            old_params = get_params(local_model_copy)

            batches = []
            counter = 0
            refresh = 0
            while counter < local_epochs:
                # Prepare batches
                if refresh == 0:
                    loader = DataLoader(DatasetSplit(dataset['train'], usersplit_train[user]), batch_size=BS, shuffle=True)
                    it = iter(loader)
                    refresh = len(loader)
                nxt_batch = next(it)
                refresh -= 1
                counter += 1
                batches.append(nxt_batch)

            #loader = DataLoader(DatasetSplit(dataset['train'], usersplit_train[user]), batch_size=BS, shuffle=True)

            for batch_idx, (inputs, labels) in enumerate(batches):
                #torch.cuda.empty_cache()

                #print(f'Local round {batch_idx}/{local_epochs - 1}')
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer = ProximalOptimizer(local_model_copy.parameters(), lr=local_lr, momentum=0.9)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = local_model_copy(inputs)
                    _, preds = torch.max(outputs, 1) # TODO Does this need keepsize=True?
                    loss = criterion(outputs, labels)
                    del outputs

                    if phase == 'train':
                        loss.backward()
                        local_params = get_params(local_model_copy)
                        optimizer.step(old_params, local_params, mu)

                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data)
                #print(f'Batch {batch_idx}: {batch_loss}, {batch_corrects}')

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                user_total_trials += len(inputs) 
                del loss

            if epoch >= num_epochs//4:
                schedulers[user].step() # Does it matter when scheduler is updated using local or global rounds?
                 
            local_weights.append(local_model_copy.state_dict()) 

            epoch_loss = running_loss / user_total_trials
            epoch_acc = running_corrects.double() / user_total_trials

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            del local_params

        w_new = aggregate(local_weights)
        model.load_state_dict(w_new)
        del local_weights
        del w_new

        if phase == 'train' and epoch_acc > best_acc:
            best_acc = epoch_acc
            #best_model_wts = copy.deepcopy(model.state_dict())

        phase = 'test'
        eval_model = copy.deepcopy(model).to(device)
        eval_model.eval() 
        running_loss = 0.0
        running_corrects = 0
        for batch_idx, (inputs, labels) in enumerate(loaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = eval_model(inputs)
                _, preds = torch.max(outputs, 1) # TODO Does this need keepsize=True?
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            batch_loss = loss.item() * inputs.size(0)
            batch_corrects = torch.sum(preds == labels.data)
            #print(f'Batch {batch_idx}: {batch_loss}, {batch_corrects}')

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        if best_test_acc < epoch_acc:
            best_test_acc = epoch_acc

        del eval_model

        tracker.add_transfer(2 * part_users * local_model_size) # Copy to and from

        tracker.add_flop(part_users * local_epochs * flops_estimate)

        tracker.add_accuracy(epoch_acc.item())

        print("cur_flops:", tracker.cur_flops/1e9)
        print("cur_transfer:", tracker.cur_transfer/1e9)
        #if tracker.cur_flops/1e9 >= 100000 or tracker.cur_transfer/1e9 >= 5000:
        #    timeout = True
        #    break # Timeout

        tracker.write(f"{status_dir}/{dataset['train'].name}_fullfedprox_dir{diric}_user{USERS}_lr{learning_rate}_frac{part_ratio}_localit{local_epochs}")


        tracker.write(f"fedprox_")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best train Acc: {best_acc:4f}')
    print(f'Best test Acc: {best_test_acc:4f}')

    #model.load_state_dict(best_model_wts)
    #del best_model_wts

    #if timeout:
    #    print("Timed out")
    tracker.write(f"{out_dir}/{dataset['train'].name}_fullfedprox_dir{diric}_user{USERS}_lr{learning_rate}_frac{part_ratio}_localit{local_epochs}")

    return model, best_test_acc, tracker.flops[-1], tracker.transferred[-1] 

# Main body

# Add scheduler?

# For Fedavg local model and global model are same
#model = MLP(INPUT_SIZE, 5000, OUTPUT_SIZE)
#model = torchvision.models.resnet18(pretrained=True)
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 200)
#model = MLPTwoHidden(1000, 10000, 10000, OUTPUT_SIZE)
#local_model = copy.deepcopy(model)
criterion = nn.CrossEntropyLoss()

#print(model)

# Refactor code from online so that train_model runs only one training round rather than all
for diric in DIRIC:
    for frac in FRAC:
        for lr in LR:
            for local_epochs in LOCAL_EPOCHS:
                model = torchvision.models.resnet18(pretrained=True)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, OUTPUT_SIZE)
                #model = MLPTwoHidden(1000, 10000, 10000, OUTPUT_SIZE)
                local_model = copy.deepcopy(model)
                criterion = nn.CrossEntropyLoss()

                print(model)

                print(f"parameters:dir-{diric},lr-{lr},local-{local_epochs},frac-{frac}")

                model, acc, flops, transferred =  train_model_federated(USERS, frac, device, model, local_model, None, criterion, MU, diric, lr, EPOCHS, local_epochs)
                print(f"acc-{acc},flops-{flops},transferred-{transferred}")

#torch.save(model, "out/fedavg_resdenseconcat_mlp_model.pt")
