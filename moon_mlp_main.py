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
from models.models import ConMLP 
from data_parsing.cifar10_data import CIFAR10Dataset
from data_parsing.cifar100_data import CIFAR100Dataset
from data_parsing.cubbirds_data_version2 import CUBBirdsDataset
from data_parsing.vggflowers_data import VGGFlowersDataset
from data_parsing.aircraft_data import AircraftDataset
from data_parsing.textures_data import TexturesDataset
from data_parsing.stanford_cars_data import StanfordCarsDataset
from data_parsing.dirichlet_sampler import *
from ist.partition import *
from models.proximal_sgd import ProximalOptimizer

from utils.plot import MetricTracker
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()

hidden_size = 1000

GPU = 2 # Change depending on others' use of GPUS
dname = args.dataset

# Parameters
parameters = {
    1000: [
        {"frac": 0.02,"loc": 1,"mu": 1, "temp": 0.5},
        {"frac": 0.02,"loc": 5,"mu": 10, "temp": 1},
        {"frac": 0.06,"loc": 1,"mu": 1, "temp": 0.5},
        {"frac": 0.06,"loc": 25,"mu": 10, "temp": 0.1}
    ],
    100: [
        {"frac": 0.1,"loc": 1,"mu": 1, "temp": 0.5},
        {"frac": 0.1,"loc": 5,"mu": 10, "temp": 1},
        #{"frac": 0.3,"loc": 1,"mu": 1, "temp": 0.5},
        #{"frac": 0.3,"loc": 25,"mu": 10, "temp": 0.1}
    ]
}

USERS=100 # Number of users
parameters = parameters[USERS]

FRAC=[0.1,0.3] # Fraction of users to report
DIRIC=[100]
#DIRIC=[0.01] # Dirichlet parameter 
LR=[0.01]
EPOCHS=1000
LOCAL_EPOCHS=[1, 5, 25]
TEMP = [0.1, 0.5, 1] # Temperature ranges in MOON paper
MU = [1, 5, 10] # Contrastive term weight in MOON paper, skip 0.1 because it is said to be more or less FedAvg
BS=32 # Initial batch size, need to adjust if is possible a batch ends up with 1 as this breaks Batch Norm

# Early cutoff threshold and sliding window
# If no improvement more than early cutoff then terminate
early_cutoff = 0.005
out_dir = f"moon_{hidden_size}_{dname}_res"
os.makedirs(f"out/{out_dir}", exist_ok=True)

class SlidingWindow:
    def __init__(self):
        self.wind_size = 50
        self.count = 0
        self.slide_wind = []

    def is_full(self):
        return self.count == self.wind_size

    def update(self, val):
        if self.count == 0 or len(self.slide_wind) == 0:
            self.slide_wind.append(val)
            self.count += 1
            return
        last = self.slide_wind[-1]
        self.count += 1
        if val > last:
            self.slide_wind.append(val)
        if self.count > self.wind_size:
            self.slide_wind.pop(0)
            self.count -= 1
        
    def get_diff(self):
        if len(self.slide_wind) == 0:
            return 0 
        if len(self.slide_wind) == 1:
            return self.slide_wind[-1]
        return self.slide_wind[-1] - self.slide_wind[0]


class SlidingWindowAvg:
    def __init__(self):
        self.wind_size = 50
        self.count = 0
        self.prev = 0
        self.slide_wind = []
    
        # Use second difference to check oscillation
        self.second_count = 0
        self.prev_diff = 0
        self.second_diff = []

    def is_full(self):
        return self.count == self.wind_size

    def update(self, val):
        diff = abs(val - self.prev)
        self.slide_wind.append(diff)
        self.prev = val
        self.count += 1
        if self.count > self.wind_size:
            self.slide_wind.pop(0)
            self.count -= 1
        
        sec_diff = abs(diff - self.prev_diff)
        self.second_diff.append(sec_diff)
        self.prev_diff = diff
        self.second_count += 1
        if self.second_count > self.wind_size - 1:
            # second window size should be one smaller 
            self.second_diff.pop(0)
            self.second_count -= 1

    def get_diff(self):
        return sum(self.slide_wind)/self.count

    def get_secdiff(self):
        return sum(self.second_diff)/(self.second_count)

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

# Alternative, just run until a certain FLOP/Byte amount? First monitor a dryrun to see a good cutoff

device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')

#feature_extractor = ResDenseConcat()
#feature_extractor.eval()

input_dim = (BS, 3072)

if dname == "cifar10":
    dataset = CIFAR10Dataset('./out/data/cifar10')
    dataset.load_features('./out/cifar10resdensefeatures.pkl')
    OUTPUT_SIZE = 10
elif dname == "cubbirds":
    dataset = CUBBirdsDataset("./data_parsing/CUB_200_2011/images_new")
    dataset.load_features('./out/cubbirdsresdensefeatures.pkl')
    OUTPUT_SIZE = 200
elif dname == "cifar100":
    dataset = CIFAR100Dataset('./out/data/cifar100')
    dataset.load_features('./out/cifar100resdensefeatures.pkl')
    OUTPUT_SIZE = 100
elif dname == "vggflowers":
    dataset = VGGFlowersDataset("./data_parsing/vggflowers/images_new")
    dataset.load_features('./out/vggflowersresdensefeatures.pkl')
    OUTPUT_SIZE = 102
elif dname == "aircraft":
    dataset = AircraftDataset("./data_parsing/aircraft/fgvc-aircraft-2013b/data/images_new")
    dataset.load_features('./out/aircraftresdensefeatures.pkl')
    OUTPUT_SIZE = 100
elif dname == "dtextures":
    dataset = TexturesDataset("./data_parsing/describe_textures/dtd//images_new")
    dataset.load_features('./out/dtexturesresdensefeatures.pkl')
    OUTPUT_SIZE = 47
elif dname == "stanfordcars":
    dataset = StanfordCarsDataset("./data_parsing/out/stanfordcars")
    dataset.load_features('./data_parsing/out/stanfordcarsresdensefeatures.pkl')
    OUTPUT_SIZE = 196

classes = dataset.classes
dataset_sizes = dataset.dataset_sizes

#loaders['train'] = [DataLoader(DatasetSplit(dataset.dataset['train'], usersplit_train[user]), batch_size=BS, shuffle=True) for user in range(USERS)]

# Implements Averaging aggregation
# Modify this for different algorithms
def aggregate(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def local_train():
    # TODO Move some of the training stuff here to make it modular 
    return

def get_params(net):
    param_group = [copy.deepcopy(t) for t in net.parameters()]
    return param_group

cos=torch.nn.CosineSimilarity(dim=-1)

def train_model_federated(num_users, part_ratio, device, model, local_model, feature_extractor, criterion, mu, temp, diric, learning_rate, num_epochs=25, local_epochs=1):
    since = time.time()
    tracker = MetricTracker(since)

    sw = SlidingWindowAvg()

    #model.to(device)
    #local_model.to(device)
    #feature_extractor.to(device)
    #feature_extractor.eval()

    #sum_model = copy.deepcopy(model)
    sum_local_model = copy.deepcopy(local_model)
    #glob_sum = summary(sum_model, input_dim, verbose=0)
    local_sum = summary(sum_local_model, input_dim, verbose=0)
    est_inputs = (torch.randn(input_dim),)
    flops = FlopCountAnalysis(local_model, est_inputs)
    flops_estimate = flops.total()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_test_acc = 0.0

    if USERS > 1:
        usersplit_train = dirichlet_sampler_idsize(dataset.dataset['train'], USERS, diric)
    else:
        # If one user use whole dataset on one node
        usersplit_train = {0: np.arange(len(dataset.dataset['train']))}

    loaders = {'test': DataLoader(dataset.dataset['test'], batch_size=BS, shuffle=True)}

    global_model_copy = copy.deepcopy(model).to(device)

    # For FedProx/FedAvg just use global model for model trained locally
    local_model_copy = copy.deepcopy(local_model).to(device)
   
    # MOON requires previous model as a comparison. Paper does not handle inconsistent participation so
    # parameters do not update every round unlike in paper but every time a user participates
    prev_parameters = [model.state_dict() for user in range(num_users)]
    prev_model = copy.deepcopy(local_model).to(device)
    for param in prev_model.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        tot_flops = 0 # Use these for bytes metric cutoffs
        tot_bytes = 0

        local_lr = learning_rate
        if epoch == int(num_epochs / 2):
            local_lr = learning_rate/10
        if epoch == int(num_epochs * 0.75):
            local_lr = learning_rate/100

        model.train()
        w_glob = model.state_dict()
        w_local = local_model.state_dict() 

        global_model_copy.load_state_dict(w_glob)

        print(f'Epoch {epoch}/{num_epochs -1}')
        print('-' * 10)
        
        local_weights = []
        local_loss = []
        
        phase = 'train' # For federated cases 'train' mode in user loop then 'test'

        # Implement user participation
        part_users= max(1, int(num_users * part_ratio)) # Minimum one user
        idxs_users = np.random.choice(range(num_users), part_users, replace=False)
        idxs_users = sorted(idxs_users) # Sort this so that printing makes sense

        # Used in IST
        #index_hidden_layer = model.partition(part_users, local_model.get_hidden_dim())
        #fc1_weight_partition = partition_FC_layer_by_output_dim_0(w_glob['fc1.weight'], index_hidden_layer)
        #bn1_weight_partition, bn1_bias_partition = partition_BN_layer(w_glob['bn1.weight'], w_glob['bn1.bias'], index_hidden_layer)
        #fc2_weight_partition = partition_FC_layer_by_input_dim_1(w_glob['fc2.weight'], index_hidden_layer)

        for u_id, user in enumerate(idxs_users):
            torch.cuda.empty_cache() 
            
            #print(f'User {user}/{num_users - 1}')

            # Used in IST
            #w_local['fc1.weight'] = fc1_weight_partition[u_id]
            #w_local['bn1.weight'] = bn1_weight_partition[u_id]
            #w_local['bn1.bias'] = bn1_bias_partition[u_id]
            #w_local['fc2.weight'] = fc2_weight_partition[u_id]
            local_params = copy.deepcopy(w_glob)
            local_model_copy.load_state_dict(local_params)

            local_model_copy.train()

            prev_model.load_state_dict(prev_parameters[user])
            prev_model.train()

            running_loss = 0.0
            running_corrects = 0
            user_total_trials = 0
            
            #local_model_copy = copy.deepcopy(local_model).to(device)

            #old_params = get_params(local_model_copy)
            
            batches = []
            counter = 0
            refresh = 0
            while counter < local_epochs:
                # Prepare batches
                if refresh == 0:
                    loader = DataLoader(DatasetSplit(dataset.dataset['train'], usersplit_train[user]), batch_size=BS, shuffle=True)
                    it = iter(loader)
                    refresh = len(loader)
                nxt_batch = next(it)
                refresh -= 1
                counter += 1
                batches.append(nxt_batch)

            #loader = DataLoader(DatasetSplit(dataset.dataset['train'], usersplit_train[user]), batch_size=BS, shuffle=True)

            #torch.cuda.empty_cache() #Does this do anything??

            optimizer = optim.SGD(local_model_copy.parameters(), lr=local_lr, momentum=0.9) 
            for batch_idx, (inputs, labels) in enumerate(batches):
                #print(f'Local round {batch_idx}/{local_epochs - 1}')
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, features = local_model_copy(inputs)
                    _, global_features = global_model_copy(inputs)
                    _, prev_features = prev_model(inputs)

                    posi = cos(features, global_features)
                    logits = posi.reshape(-1,1)
                    nega = cos(features, prev_features)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
                    logits /= temp

                    targets = torch.zeros(inputs.size(0)).to(device).long()
                    loss2 = mu * criterion(logits, targets)

                    _, preds = torch.max(outputs, 1) # TODO Does this need keepsize=True?
                    loss1 = criterion(outputs, labels)
                    loss = loss1 + loss2

                    if phase == 'train':
                        loss.backward()
                        #local_params = get_params(local_model_copy)
                        optimizer.step()

                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data)
                #print(f'Batch {batch_idx}: {batch_loss}, {batch_corrects}') # Debug?

                prev_parameters[user] = local_model_copy.state_dict()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                user_total_trials += len(inputs) 

            #if epoch >= num_epochs//4:
            #    schedulers[user].step() # Does it matter when scheduler is updated using local or global rounds?
                    
            local_weights.append(copy.deepcopy(local_model_copy.state_dict())) 

            del local_params

            epoch_loss = running_loss / user_total_trials
            epoch_acc = running_corrects.double() / user_total_trials

            #print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        w_new = aggregate(local_weights)
        model.load_state_dict(w_new)

        if phase == 'train' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

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
                outputs, _ = eval_model(inputs)
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

        tracker.add_transfer(2 * part_users * local_sum.total_param_bytes) # Copy to and from

        tracker.add_flop(2 * part_users * local_epochs * flops_estimate) # Additional two forward propagations leads to doubled flops

        tracker.add_accuracy(epoch_acc.item())

        sw.update(epoch_acc)
        #print(sw.get_diff())
        #if sw.is_full() and sw.get_diff() < early_cutoff:
        #    break

        if epoch % 10 == 9:
            tracker.write(f"temp_plot_gpu{GPU}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best train Acc: {best_acc:4f}')
    print(f'Best test Acc: {best_test_acc:4f}')

    model.load_state_dict(best_model_wts)

    tracker.write(f"{out_dir}/{dataset.dataset['train'].name}_moon_dir{diric}_user{num_users}_lr{lr}_frac{part_ratio}_localit{local_epochs}_mu{mu}_temp{temp}")

    return model, best_test_acc, tracker.flops[-1], tracker.transferred[-1] 

# Main body
#participating_users= max(1, int(USERS * FRAC)) # Minimum one user

# For Fedavg local model and global model are same
criterion = nn.CrossEntropyLoss()

for diric in DIRIC:
    for lr in LR:
        for p in parameters:
            frac = p['frac']
            local_epochs = p['loc']
            mu = p['mu']
            temp = p['temp']
            local_hidden_size = hidden_size # Same model for FedAvg

            model = ConMLP(3072, hidden_size, OUTPUT_SIZE)
            local_model = ConMLP(3072, local_hidden_size, OUTPUT_SIZE)

            print(model)
            print(local_model)

            print(f"parameters:dir-{diric},lr-{lr},local-{local_epochs},frac-{frac},temp-{temp},mu-{mu}")
            model, acc, flops, transferred =  train_model_federated(USERS, frac, device, model, local_model, None, criterion, mu, temp, diric, lr, EPOCHS, local_epochs)
            print(f"acc-{acc},flops-{flops},transferred-{transferred}")

#torch.save(model, "out/fedist_resdenseconcat_mlp_model.pt")
