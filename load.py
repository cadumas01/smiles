from video_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms 
import os
import PIL
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

# Returns data loaders
# Returns train_loader, validation_loader
def get_loaders(dataset, batch_size, validation_split, num_workers=2, pin_memory=False):
    # https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))


    # shuffle dataset
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # https://pytorch.org/docs/stable/data.html
    # convert to more workable from
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, num_workers=num_workers, pin_memory=pin_memory)


    return train_loader, validation_loader    
