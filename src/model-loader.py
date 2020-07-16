import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import  torchvision.datasets
import torchvision.models as models
from model import *

alexnet = models.alexnet(pretrained=True)
img_path = "../dataset/" # location to the image path.
image_dataset = torchvision.datasets.ImageFolder(img_path, transform=torchvision.transforms.ToTensor())


model_load = MLP()
model_data = torch.load('../CNN') # location to the save CNN using the checkpoint
model_load.load_state_dict(model_data)


def get_accuracy(model, data):
    img = data
    features = alexnet.features(img.unsqueeze(0)).detach()

    model.eval() # annotate model for evaluation
    output = model(features) # We don't need to run torch.softmax
    pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    pred = pred.detach().numpy()
    return ("Tumor Detected" if (pred == 1).sum() else "No Tumor Detected")



for i in range(len(image_dataset)):
  img, label = image_dataset[i] # Only take the first image and it's label
  print(get_accuracy(model_load, img)) # run get accuracy on the above 1 image