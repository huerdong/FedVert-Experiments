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

from feature_extractors.feature_extractors import ResDenseConcat
from models.models import ServerMLP, DeviceMLP
from data_parsing.cifar10_data import CIFAR10Dataset
from data_parsing.dirichlet_sampler import dirichlet_sampler
from ist.partition import *

LR=0.0125
BS=32

dataset = CIFAR10Dataset('./out/data/cifar10')
classes = dataset.classes
dataset_sizes = dataset.dataset_sizes

USERS= 10 # Number of users
FRAC=1 # Fraction of users to report
DIRIC= 10 # Dirichlet parameter 

usersplit_train = dirichlet_sampler(dataset.dataset['train'], USERS, DIRIC)

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

# Dataloaders will still be better off handled here instead of in data modules
loaders = {'test': DataLoader(dataset.dataset['test'], batch_size=BS, shuffle=True)}
loaders['train'] = [DataLoader(DatasetSplit(dataset.dataset['train'], usersplit_train[user]), batch_size=BS, shuffle=True) for user in range(USERS)]

GPU = 1 # Change depending on others' use of GPUS
device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')

# Implements IST aggregation
# Modify this for different algorithms
def aggregate(w_glob, w, fc1_index):
    new_server_glob = copy.deepcopy(w_glob)

    layer_fc1_weight = []
    layer_bn1_weight = []
    layer_bn1_bias = []
    layer_fc2_weight = []

    for i in range(len(w)):
        layer_fc1_weight.append(w[i]['fc1.weight'].cpu())
        layer_bn1_weight.append(w[i]['bn1.weight'].cpu())
        layer_bn1_bias.append(w[i]['bn1.bias'].cpu())
        layer_fc2_weight.append(w[i]['fc2.weight'].cpu())

    update_tensor_by_update_lists_dim_0(new_server_glob['fc1.weight'], layer_fc1_weight, fc1_index)
    update_tensor_by_update_lists_dim_0(new_server_glob['bn1.weight'], layer_bn1_weight, fc1_index)
    update_tensor_by_update_lists_dim_0(new_server_glob['bn1.bias'], layer_bn1_bias, fc1_index)
    update_tensor_by_update_lists_dim_1(new_server_glob['fc2.weight'], layer_fc2_weight, fc1_index)

    return new_server_glob

def local_train():
    # TODO Move some of the training stuff here to make it modular 
    return

# Add scheduler?
def train_model_federated(num_users, part_ratio, device, model, local_model, feature_extractor, criterion, num_epochs=25, local_epochs=1):
    since = time.time()
    #model.to(device)
    #local_model.to(device)
    feature_extractor.to(device)
    feature_extractor.eval()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        w_glob = model.state_dict()
        w_local = local_model.state_dict()

        print(f'Epoch {epoch}/{num_epochs -1}')
        print('-' * 10)
        
        local_weights = []
        local_loss = []
        
        phase = 'train' # For federated cases 'train' mode in user loop then 'test'

        # Implement user participation
        part_users= max(1, num_users * part_ratio) # Minimum one user
        idxs_users = np.random.choice(range(num_users), part_users, replace=False)
        idxs_users = sorted(idxs_users) # Sort this so that printing makes sense

        index_hidden_layer = model.partition(part_users, local_model.get_hidden_dim())
        fc1_weight_partition = partition_FC_layer_by_output_dim_0(w_glob['fc1.weight'], index_hidden_layer)
        bn1_weight_partition, bn1_bias_partition = partition_BN_layer(w_glob['bn1.weight'], w_glob['bn1.bias'], index_hidden_layer)
        fc2_weight_partition = partition_FC_layer_by_input_dim_1(w_glob['fc2.weight'], index_hidden_layer)

        for user in idxs_users:
            print(f'User {user}/{num_users - 1}')

            w_local['fc1.weight'] = fc1_weight_partition[user]
            w_local['bn1.weight'] = bn1_weight_partition[user]
            w_local['bn1.bias'] = bn1_bias_partition[user]
            w_local['fc2.weight'] = fc2_weight_partition[user]
            local_model.load_state_dict(w_local)

            local_model.train()

            running_loss = 0.0
            running_corrects = 0
            
            local_model.load_state_dict(copy.deepcopy(w_local))
            user_total_trials = 0

            local_model_copy = copy.deepcopy(local_model).to(device)

            for local_ep in range(local_epochs):
                print(f'Local round {local_ep}/{local_epochs - 1}')
                
                for batch_idx, (inputs, labels) in enumerate(loaders[phase][user]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # TODO Move optimizer to global, create separate optimizer instances per client
                    # Otherwise scheduler wouldn't work
                    optimizer = optim.SGD(local_model_copy.parameters(), lr=LR, momentum=0.9)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = local_model_copy(feature_extractor, inputs)
                        _, preds = torch.max(outputs, 1) # TODO Does this need keepsize=True?
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    batch_loss = loss.item() * inputs.size(0)
                    batch_corrects = torch.sum(preds == labels.data)
                    print(f'Batch {batch_idx}: {batch_loss}, {batch_corrects}')

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    user_total_trials += len(inputs) 

            local_weights.append(copy.deepcopy(local_model_copy.state_dict())) 

            epoch_loss = running_loss / user_total_trials
            epoch_acc = running_corrects.double() / user_total_trials

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        w_new = aggregate(w_glob, local_weights, index_hidden_layer)
        model.load_state_dict(w_new)

        if phase == 'train' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        eval_model = copy.deepcopy(model).to(device)
        phase = 'test'
        eval_model.eval() 
        running_loss = 0.0
        running_corrects = 0
        for batch_idx, (inputs, labels) in enumerate(loaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = eval_model(feature_extractor, inputs)
                _, preds = torch.max(outputs, 1) # TODO Does this need keepsize=True?
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            batch_loss = loss.item() * inputs.size(0)
            batch_corrects = torch.sum(preds == labels.data)
            print(f'Batch {batch_idx}: {batch_loss}, {batch_corrects}')

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best train Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# Main body
feature_extractor = ResDenseConcat()
feature_extractor.eval()

# Add scheduler?

# For Fedavg local model and global model are same
model = ServerMLP(2000, 5000, 10)
# Check with someone (Yuxin?) about how to handle size of client model TODO
# Should it be smaller, bigger, or equal to hidden layer/users
local_model = DeviceMLP(2000, 500, 10)
criterion = nn.CrossEntropyLoss()

print(model)

EPOCHS=10
LOCAL_EPOCHS=1

# Refactor code from online so that train_model runs only one training round rather than all
model =  train_model_federated(USERS, FRAC, device, model, local_model, feature_extractor, criterion, EPOCHS, LOCAL_EPOCHS)
torch.save(model, "out/fedavg_resdenseconcat_mlp_model.pt")
