import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import  torchvision.datasets
import torchvision.models as models


# %matplotlib inline

# Mount google drive to fetch data

# from google.colab import drive
# drive.mount('/content/gdrive')

# Load alexnet network
alexnet = models.alexnet(pretrained=True)


def fetchData():
    '''
    fetch data from file.
    @return: train, test and valid dataset
    '''
    train_path = "../dataset/" # edit me
    # valid_path = "" # edit me
    # test_path = ""   # edit me

    train_data = torchvision.datasets.ImageFolder(train_path, transform=torchvision.transforms.ToTensor())
    # valid_data = torchvision.datasets.ImageFolder(valid_path, transform=torchvision.transforms.ToTensor())
    # test_data = torchvision.datasets.ImageFolder(test_path, transform=torchvision.transforms.ToTensor())
    # return train_data, valid_data, test_data
    return train_data


train_data = fetchData()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)

train_data_features = []
i = 0
for img, y in train_data:
  features = alexnet.features(img.unsqueeze(0)).detach()  # compute the alex net features based on the image
  train_data_features.append((features, y),)

