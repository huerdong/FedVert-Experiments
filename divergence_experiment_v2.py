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
from models.models import MLP, ServerMLP, DeviceMLP
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

import statistics
import argparse

train_central = True

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()

GPU1 = 2 # Change depending on others' use of GPUS
GPU2 = 3
dname = args.dataset

out_dir = f"out/divergence_{dname}_v2"
os.makedirs(out_dir, exist_ok=True)

model_dir = f"out/divergence_model_{dname}"
os.makedirs(model_dir, exist_ok=True)

# These aren't in ranges because we are certain we use these
USERS=100 # Number of users
FRAC={
    100: 0.1
}
FRAC = FRAC[USERS]
#DIRIC=[100]
DIRIC=0.01 # Dirichlet parameter 
LR=0.01
#LR=[0.1, 0.01, 0.001]
#LOCAL_EPOCHS=[1]

algs = ["fedavg", "fedprox", "ist", "istprox"]
#algs = ["ist", "istprox"]

alg_local_ep = {"fedavg": 25, "fedprox": 5, "ist": 5, "istprox": 5}

mu = 0.2

BS=32 # Initial batch size, need to adjust if is possible a batch ends up with 1 as this breaks Batch Norm
EPOCHS=100

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

device_avg = torch.device('cuda:{}'.format(GPU1) if torch.cuda.is_available() else 'cpu')
device_ist = torch.device('cuda:{}'.format(GPU2) if torch.cuda.is_available() else 'cpu')

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

def w_delta(w1, w2):
    aggregate = None
    for k in w1.keys(): # Assume keys are same
        if aggregate is None:
            aggregate = torch.linalg.norm(torch.sub(w1[k], w2[k])).square()
        else:
            aggregate += torch.linalg.norm(torch.sub(w1[k], w2[k])).square()
    aggregate = aggregate.sqrt()
    return aggregate.item()

def w_dir(w1, w2):
    norm = w_delta(w1, w2)
    aggregate = {}
    for k in w1.keys():
        if norm < 1e-5:
            aggregate[k] = torch.as_tensor(0.0)
        else:
            aggregate[k] = torch.sub(w1[k], w2[k]).div(norm)
    return aggregate

def w_norm(w):
    aggregate = None
    for k in w.keys(): # Assume keys are same
        if aggregate is None:
            aggregate = torch.linalg.norm(w[k]).square()
        else:
            aggregate += torch.linalg.norm(w[k]).square()
    aggregate = aggregate.sqrt()
    return aggregate.item()

def cosine_sim(w1, w2):
    dot_prod = None
    for k in w1.keys():
        if dot_prod is None:
            dot_prod = torch.sum(torch.mul(w1[k], w2[k]))
        else:
            dot_prod += torch.sum(torch.mul(w1[k], w2[k]))
    sim = dot_prod.div(w_norm(w1)).div(w_norm(w2))
    return sim.item()

def avg_pairwise_cosine_sim(w_loc):
    n = len(w_loc)
    cosines = []
    for i in range(n):
        for j in range(i):
            cosines.append(cosine_sim(w_loc[i], w_loc[j]))
    mu = statistics.mean(cosines)
    sigma = statistics.pstdev(cosines)
    return mu, sigma

def aggregate_avg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def aggregate_ist(w_glob, w, fc1_index):
    new_server_glob = w_glob

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

def get_params(net):
    param_group = [copy.deepcopy(t) for t in net.parameters()]
    return param_group

def central_train(device, model, data, optimizer):
    model.to(device)
    model.train()
    
    loader = DataLoader(DatasetSplit(dataset.dataset['train'], data), batch_size=BS, shuffle=True)

    #loader = torch.utils.data.DataLoader(data, batch_size=BS,
    #                                      shuffle=True, num_workers=2)
    # Run for an entire epoch
    for (inputs, labels) in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def local_train(device, model, alg, rounds, dataset, loc_data):
    model.to(device)
    model.train()

    if alg in ["fedprox", "istprox"]:
        optimizer = ProximalOptimizer(model.parameters(), lr=LR, momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)    

    batches = []
    counter = 0
    refresh = 0
    while counter < rounds:
        # Prepare batches
        if refresh == 0:
            loader = DataLoader(DatasetSplit(dataset.dataset['train'], loc_data), batch_size=BS, shuffle=True)
            it = iter(loader)
            refresh = len(loader)
        nxt_batch = next(it)
        refresh -= 1
        counter += 1
        batches.append(nxt_batch)

    phase = 'train'
    old_params = get_params(model)
    for (inputs, labels) in batches:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) # TODO Does this need keepsize=True?
            loss = criterion(outputs, labels)
            if phase == 'train':
                loss.backward()
                if alg in ["fedprox", "istprox"]:
                    local_params = get_params(model)
                    optimizer.step(old_params, local_params, mu)
                else: 
                    optimizer.step()
    return copy.deepcopy(model.state_dict())

criterion = nn.CrossEntropyLoss()

avg_hidden_size = 1000
ist_hidden_size = 3000
ist_loc_size = 300

avg_model = MLP(3072, avg_hidden_size, OUTPUT_SIZE).to(device_avg)
ist_model = ServerMLP(3072, ist_hidden_size, OUTPUT_SIZE) #.to(device_ist)
local_model = DeviceMLP(3072, ist_loc_size, OUTPUT_SIZE).to(device_ist)

central_model_avg = copy.deepcopy(avg_model).to(device_avg)
central_model_ist = copy.deepcopy(ist_model).to(device_ist)

central_optimizer_avg = optim.SGD(central_model_avg.parameters(), lr=LR, momentum=0.9)
central_optimizer_ist = optim.SGD(central_model_ist.parameters(), lr=LR, momentum=0.9)

models = {
    "fedavg": copy.deepcopy(avg_model).to(device_avg),
    "fedprox": copy.deepcopy(avg_model).to(device_avg),
    "ist": copy.deepcopy(ist_model), #.to(device_ist),
    "istprox": copy.deepcopy(ist_model) #.to(device_ist)
}

delta_results = {v: [] for v in algs}
cosine_results = {"fedavg": [], "fedprox": []}

def write_results(data, alg, type, out=out_dir):
    fname = f"{out}/{alg}_{type}.csv"
    with open(fname, "w") as ofile:
        for idx, v in enumerate(data):
            ofile.write(f"{idx},{v}\n")

def eval(model, dataset, device):
    eval_model = copy.deepcopy(model).to(device)
    eval_model.eval()
    loader = DataLoader(dataset, batch_size=BS, shuffle=True)

    running_loss = 0
    running_corrects = 0
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = eval_model(inputs)
        _, preds = torch.max(outputs, 1) # TODO Does this need keepsize=True?
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['test']
    epoch_acc = running_corrects.double() / dataset_sizes['test']
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_acc

central_avg_acc = []
central_ist_acc = []

for e in range(EPOCHS):
    print(f"Epoch {e}:")
    usersplit_train = dirichlet_sampler_idsize(dataset.dataset['train'], USERS, DIRIC)
    #data_central = dataset.dataset['train']

    part_users= max(1, int(USERS * FRAC)) # Minimum one user
    idxs_users = np.random.choice(range(USERS), part_users, replace=False)
    idxs_users = sorted(idxs_users) # Sort this so that printing makes sense

    data_central = [v for u in idxs_users for v in usersplit_train[u]]

    w_central_avg_init = copy.deepcopy(central_model_avg.state_dict())
    w_central_ist_init = copy.deepcopy(central_model_ist.state_dict())

    # Initialize all algorithms to be same as central models to get comparison of deviations
    for alg in algs:
        if "ist" in alg:
            models[alg].load_state_dict(w_central_ist_init)
            #pass
        else:
            models[alg].load_state_dict(w_central_avg_init)

    if central_train:
        central_train(device_avg, central_model_avg, data_central, central_optimizer_avg)
        central_train(device_ist, central_model_ist, data_central, central_optimizer_ist)

        torch.save(central_model_avg.state_dict(), f"{model_dir}/{e}_avg_model.pt")
        torch.save(central_model_ist.state_dict(), f"{model_dir}/{e}_ist_model.pt")
    else:
        # Do we need to load the optimizer as well? No because in this case we only need the weights for experiments
        central_model_avg.load_state_dict(torch.load(f"{model_dir}/{e}_avg_model.pt"))
        central_model_ist.load_state_dict(torch.load(f"{model_dir}/{e}_ist_model.pt"))

    w_central_avg = central_model_avg.state_dict()
    w_central_ist = central_model_ist.state_dict()

    w_central_avg_dir = w_dir(w_central_avg_init, w_central_avg)
    w_central_ist_dir = w_dir(w_central_ist_init, w_central_ist)

    print("Central for average algs")
    avg_acc = eval(central_model_avg, dataset.dataset['test'], device_avg)
    print("Central for ist algs")
    ist_acc = eval(central_model_ist, dataset.dataset['test'], device_ist)
    central_avg_acc.append(avg_acc)
    central_ist_acc.append(ist_acc)

    write_results(central_avg_acc, "central_avg", "acc")
    write_results(central_ist_acc, "central_ist", "acc")

    for alg in algs:
        if train_central:
            continue
        print(f"Algorithm {alg}")
        print("------------------------")
        model = models[alg]

        w_glob = copy.deepcopy(model.state_dict())
        if "ist" in alg:
            # Used in IST
            index_hidden_layer = model.partition(part_users, local_model.get_hidden_dim())
            fc1_weight_partition = partition_FC_layer_by_output_dim_0(w_glob['fc1.weight'], index_hidden_layer)
            bn1_weight_partition, bn1_bias_partition = partition_BN_layer(w_glob['bn1.weight'], w_glob['bn1.bias'], index_hidden_layer)
            fc2_weight_partition = partition_FC_layer_by_input_dim_1(w_glob['fc2.weight'], index_hidden_layer)

        local_weights = []
        for u_id, user in enumerate(idxs_users):
            loc_data = usersplit_train[u_id]

            if "ist" in alg: 
                local_model_copy = copy.deepcopy(local_model).to(device_ist)
                w_local = copy.deepcopy(local_model.state_dict())
                w_local['fc1.weight'] = fc1_weight_partition[u_id]
                w_local['bn1.weight'] = bn1_weight_partition[u_id]
                w_local['bn1.bias'] = bn1_bias_partition[u_id]
                w_local['fc2.weight'] = fc2_weight_partition[u_id]
                local_model_copy.load_state_dict(w_local)
                
                weights = local_train(device_ist, local_model_copy, alg, alg_local_ep[alg], dataset, loc_data)
            else:
                local_model_copy = copy.deepcopy(model).to(device_avg)

                weights = local_train(device_avg, local_model_copy, alg, alg_local_ep[alg], dataset, loc_data)

            local_weights.append(weights) 

        if "ist" in alg:
            w_central_dir = w_central_ist_dir
            w_agg = aggregate_ist(w_glob, local_weights, index_hidden_layer)
        else:
            w_central_dir = w_central_avg_dir
            w_agg = aggregate_avg(local_weights)

            cos = avg_pairwise_cosine_sim(local_weights) 
            cosine_results[alg].append(cos)
            print(f"Avg Pairwise Cosine Sim: {cos}")

            write_results(cosine_results[alg], alg, "pair_cosine")

        agg_dir = w_dir(w_agg, w_glob)
        if "ist" in alg:
            for k in agg_dir.keys():
                agg_dir[k] = agg_dir[k].to(device_ist)
        delta = cosine_sim(agg_dir, w_central_dir)
        delta_results[alg].append(delta)

        print(f"Normalized directional deviation: {delta}")

        write_results(delta_results[alg], alg, "dir_cosine")
