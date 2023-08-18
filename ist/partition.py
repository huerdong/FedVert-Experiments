# some functions come from Binhang
import torch
from torch import nn
import torch.nn.functional as F
from random import shuffle

def partition_BN_layer(weight_tensor, bias_tensor, index_buffer):
    weight_dim0 = weight_tensor.shape[0]
    bias_dim0 = bias_tensor.shape[0]
    split_num = len(index_buffer)
    #assert (weight_dim0 == bias_dim0 and weight_dim0 % split_num == 0)
    weight_results = []
    bias_results = []
    for current_indexes in index_buffer:
        current_weight_tensor = torch.index_select(weight_tensor, 0, current_indexes)
        current_bias_tensor = torch.index_select(bias_tensor, 0, current_indexes)
        weight_results.append(current_weight_tensor)
        bias_results.append(current_bias_tensor)
    return weight_results, bias_results


def partition_FC_layer_by_output_dim_0(tensor, index_buffer):
    weight_dim0 = tensor.shape[0]
    split_num = len(index_buffer)
    #assert(weight_dim0 % split_num == 0)
    results = []
    for current_indexes in index_buffer:
        current_weight_tensor = torch.index_select(tensor, 0, current_indexes)
        results.append(current_weight_tensor)
    return results

def partition_FC_layer_by_input_dim_1(tensor, index_buffer):
    weight_dim1 = tensor.shape[1]
    split_num = len(index_buffer)
    #assert(weight_dim1 % split_num == 0)
    results = []
    for current_indexes in index_buffer:
        current_weight_tensor = torch.index_select(tensor, 1, current_indexes)
        results.append(current_weight_tensor)
    return results

def partition_FC_layer_by_dim_01(tensor, index_buffer0, index_buffer1):
    '''
    :param tensor:
    :param index_buffer0: dim 0 is the output dim
    :param index_buffer1: dim 1 is the input dim
    :return:
    '''
    dim0 = tensor.shape[0]
    dim1 = tensor.shape[1]
    #assert(len(index_buffer0)==len(index_buffer1))
    split_num = len(index_buffer0)
    #assert (dim0 % split_num == 0 and dim1 % split_num == 0)
    results = []
    for i in range(split_num):
        temp_tensor = torch.index_select(tensor, 0, index_buffer0[i])
        current_tensor = torch.index_select(temp_tensor, 1, index_buffer1[i])
        results.append(current_tensor)
    return results

def update_tensor_by_update_lists_dim_0(tensor, update_list, index_buffer):
    #assert(len(update_list) == len(index_buffer))
    for i in range(len(update_list)):
        tensor.index_copy_(0, index_buffer[i], update_list[i])

def update_tensor_by_update_lists_dim_1(tensor, update_list, index_buffer):
    #assert(len(update_list) == len(index_buffer))
    for i in range(len(update_list)):
        tensor.index_copy_(1, index_buffer[i], update_list[i])

def update_tensor_by_update_lists_dim_01(tensor, update_list, index_buffer0, index_buffer1):
    #assert(len(update_list) == len(index_buffer0)
    #       and len(update_list) == len(index_buffer1))
    for i in range(len(update_list)):
        temp_tensor = torch.index_select(tensor, 0, index_buffer0[i])
        temp_tensor.index_copy_(1, index_buffer1[i], update_list[i])
        tensor.index_copy_(0, index_buffer0[i], temp_tensor)

