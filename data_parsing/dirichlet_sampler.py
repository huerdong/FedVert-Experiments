#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from torchvision import datasets, transforms
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

# TODO Create two versions one that replaces and one that does not (This one)
def dirichlet_sampler(dataset, num_users, diric=100):
    dataset_size = len(dataset)
    num_classes = len(dataset.classes)
    
    prior_distribution = [1 for i in range(num_users)]
    distributions = np.random.dirichlet(diric * np.array(prior_distribution), num_classes)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    items_per_class = dict(Counter(dataset.targets))
    idxs = np.arange(dataset_size)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    hsv = plt.get_cmap('hsv')
    color_num = num_classes
    plot_colors = hsv(np.linspace(0, 1.0, color_num))
    space = [0.0 for i in range(num_users)]
    for i in range(num_classes):
        plt.barh(range(num_users), distributions[i], left=space, color=plot_colors[i])
        space += distributions[i]
    plt.savefig(f'./out/{type(dataset).__name__}_users{num_users}_dir{diric}_distribution.png')

    for i in range(num_classes):
        idxs_set = idxs_labels[0, np.where(idxs_labels[1, :] == i)][0].tolist()
        images_distribution = (items_per_class[i] * distributions[i]).astype(int)
        temp_total = sum(images_distribution) # Check if we lost any
        images_distribution[-1] += items_per_class[i] - temp_total # Add any leftovers to last entry
        for j in range(num_users):
            #print(f"User {j} has chosen {images_distribution[j]} samples for class {i}")
            dict_users[j] = np.concatenate((dict_users[j], np.random.choice(idxs_set, images_distribution[j], replace=False)), axis=0)

    return dict_users

def dirichlet_sampler_idsize(dataset, num_users, diric=100, items_per_user=500):
    # TODO Make items_per_user also inputable as an array for non-identical quantity as well
    dataset_size = len(dataset)
    num_classes = len(dataset.classes)
    
    # Assume data is initially even distributed
    prior_distribution = [1 for i in range(num_classes)]
    distributions = np.random.dirichlet(diric * np.array(prior_distribution), num_users).transpose()

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    #items_per_user = dataset_size / num_users # Make sure this is an integer?

    idxs = np.arange(dataset_size)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    hsv = plt.get_cmap('hsv')
    color_num = num_classes
    plot_colors = hsv(np.linspace(0, 1.0, color_num))
    space = [0.0 for i in range(num_users)]
    for i in range(num_classes):
        plt.barh(range(num_users), distributions[i], left=space, color=plot_colors[i])
        space += distributions[i]
    plt.savefig(f'./out/{dataset.name}_users{num_users}_dir{diric}_distribution_idsize.png')

    distributions = distributions.transpose()

    for i in range(num_users):
        images_distribution = np.round((items_per_user * distributions[i])).astype(int)
        images_distribution[-1] = max(items_per_user - sum(images_distribution[0:-1]), 0) # Maybe we'll get extras but it is fine
        for j in range(len(images_distribution)):
            idxs_set = idxs_labels[0, np.where(idxs_labels[1, :] == j)][0].tolist()
            #print(f"User {i} has chosen {images_distribution[j]} samples for class {j}")
            dict_users[i] = np.concatenate((dict_users[i], np.random.choice(idxs_set, images_distribution[j], replace=True)), axis=0)
    
    return dict_users