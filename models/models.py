import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle

# Experimentally we determined that best performance is with BN after first layer
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.tag = "MLP"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        # TODO Check Affine True/False performance
        self.bn1 = nn.BatchNorm1d(self.hidden_size, momentum=1.0, affine=True, track_running_stats=False)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        #self.leaky = nn.LeakyReLU(negative_slope=0.01) I don't think leakyrelu does much useful

    def get_size(self):
        return 8 * (self.input_size * self.hidden_size + self.hidden_size * self.output_size + 2 * self.hidden_size) # 8 bytes from 64-bit floats

    # Is forward and backward same amount?
    def forward_flops(self, data_size):
        # Ignore batch norm?
        return data_size * (self.input_size * self.hidden_size + self.hidden_size * self.output_size)

    def backward_flops(self, data_size):
        return data_size * (self.input_size * self.hidden_size + self.hidden_size * self.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        #x = self.leaky(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

# Version that also outputs feature vectors for MOON
class ConMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ConMLP, self).__init__()
        self.tag = "ConMLP"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        # TODO Check Affine True/False performance
        self.bn1 = nn.BatchNorm1d(self.hidden_size, momentum=1.0, affine=True, track_running_stats=False)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        #self.leaky = nn.LeakyReLU(negative_slope=0.01) I don't think leakyrelu does much useful

    def get_size(self):
        return 8 * (self.input_size * self.hidden_size + self.hidden_size * self.output_size + 2 * self.hidden_size) # 8 bytes from 64-bit floats

    # Is forward and backward same amount?
    def forward_flops(self, data_size):
        # Ignore batch norm?
        return data_size * (self.input_size * self.hidden_size + self.hidden_size * self.output_size)

    def backward_flops(self, data_size):
        return data_size * (self.input_size * self.hidden_size + self.hidden_size * self.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        y = F.relu(x, inplace=True)
        #x = self.leaky(x)
        x = self.fc2(y)
        x = F.log_softmax(x, dim=1)
        return x, y 

class ConMLPTwoHidden(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ConMLPTwoHidden, self).__init__()
        self.tag = "ConMLPTwoHidden"
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.bn1 = nn.BatchNorm1d(self.hidden_size1, momentum=1.0, affine=True, track_running_stats=True)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size1, momentum=1.0, affine=True, track_running_stats=True)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = self.bn2(x)
        y = F.relu(x, inplace=True)
        x = self.fc3(y)
        x = F.log_softmax(x, dim=1)
        return x, y

    def get_size(self):
        return 8 * (self.input_size * self.hidden_size1 + self.hidden_size1 * self.hidden_size2 + self.hidden_size2 * self.output_size + 2 * self.hidden_size1 + 2 * self.hidden_size2) # 8 bytes from 64-bit floats

    # Is forward and backward same amount?
    def forward_flops(self, data_size):
        # Ignore batch norm?
        return data_size * (self.input_size * self.hidden_size1 + self.hidden_size1 * self.hidden_size2 + self.hidden_size2 * self.output_size)

    def backward_flops(self, data_size):
        return data_size * (self.input_size * self.hidden_size1 + self.hidden_size1 * self.hidden_size2 + self.hidden_size2 * self.output_size)

# Experimentally we determined that best performance is with BN after first layer
class MLPwithBN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPwithBN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size, momentum=1.0, affine=True, track_running_stats=False)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        #self.bn2 = nn.BatchNorm1d(self.output_size, momentum=1.0, affine=True, track_running_states=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.log_softmax(x, dim=1)
        return x

class MLPTwoHidden(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLPTwoHidden, self).__init__()
        self.tag = "MLPTwoHidden"
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.bn1 = nn.BatchNorm1d(self.hidden_size1, momentum=1.0, affine=True, track_running_stats=True)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size2, momentum=1.0, affine=True, track_running_stats=True)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def get_size(self):
        return 8 * (self.input_size * self.hidden_size1 + self.hidden_size1 * self.hidden_size2 + self.hidden_size2 * self.output_size + 2 * self.hidden_size1 + 2 * self.hidden_size2) # 8 bytes from 64-bit floats

    # Is forward and backward same amount?
    def forward_flops(self, data_size):
        # Ignore batch norm?
        return data_size * (self.input_size * self.hidden_size1 + self.hidden_size1 * self.hidden_size2 + self.hidden_size2 * self.output_size)

    def backward_flops(self, data_size):
        return data_size * (self.input_size * self.hidden_size1 + self.hidden_size1 * self.hidden_size2 + self.hidden_size2 * self.output_size)

class ServerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ServerMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size, momentum=1.0, affine=True, track_running_stats=False)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        self.partition_index = [i for i in range(self.hidden_size)]

    def partition(self, device_num, device_hidden_layer_size):
        index_hidden_layer = []
        
        shuffle(self.partition_index)
        for i in range(device_num):
            index = torch.tensor(self.partition_index[i * device_hidden_layer_size: (i+1) * device_hidden_layer_size])
            index_hidden_layer.append(index)

        return index_hidden_layer
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class DeviceMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeviceMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size, momentum=1.0, affine=True, track_running_stats=False)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def get_hidden_dim(self):
        return self.hidden_size

    def get_size(self):
        return 8 * (self.input_size * self.hidden_size + self.hidden_size * self.output_size + 2 * self.hidden_size) # 8 bytes from 64-bit floats

    # Is forward and backward same amount?
    def forward_flops(self, data_size):
        # Ignore batch norm?
        return data_size * (self.input_size * self.hidden_size + self.hidden_size * self.output_size)

    def backward_flops(self, data_size):
        return data_size * (self.input_size * self.hidden_size + self.hidden_size * self.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class ServerMLPTwoHidden(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ServerMLPTwoHidden, self).__init__()
        self.tag = "MLPTwoHidden"
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.bn1 = nn.BatchNorm1d(self.hidden_size1, momentum=1.0, affine=True, track_running_stats=True)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size2, momentum=1.0, affine=True, track_running_stats=True)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

        self.partition_index1 = [i for i in range(self.hidden_size1)]
        self.partition_index2 = [i for i in range(self.hidden_size2)]

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def partition(self, device_num, device_hidden_layer_size1, device_hidden_layer_size2):
        index_hidden_layer1 = []
        index_hidden_layer2 = []
        
        shuffle(self.partition_index1)
        shuffle(self.partition_index2)

        for i in range(device_num):
            # get the index for partition hidden layer 1
            Index1 = torch.tensor(
                self.partition_index1[i * device_hidden_layer_size1 : (i + 1) * device_hidden_layer_size1])
            index_hidden_layer1.append(Index1)

            # get the index for partition hidden layer 2
            Index2 = torch.tensor(
                self.partition_index2[i * device_hidden_layer_size2 : (i + 1) * device_hidden_layer_size2])
            index_hidden_layer2.append(Index2)

        return index_hidden_layer1, index_hidden_layer2
    

class DeviceMLPTwoHidden(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(DeviceMLPTwoHidden, self).__init__()
        self.tag = "MLPTwoHidden"
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.bn1 = nn.BatchNorm1d(self.hidden_size1, momentum=1.0, affine=True, track_running_stats=True)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size2, momentum=1.0, affine=True, track_running_stats=True)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def get_hidden_dim(self):
        return self.hidden_size1, self.hidden_size2