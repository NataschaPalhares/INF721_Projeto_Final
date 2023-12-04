import os
import torch
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix



class CNN(torch.nn.Module):
  def __init__(self, num_classes):
    super(CNN, self).__init__()

    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=1)
    self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=1)
    
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    
    self.batch_norm1 = torch.nn.BatchNorm2d(64)
    self.batch_norm2 = torch.nn.BatchNorm2d(128)
    self.batch_norm3 = torch.nn.BatchNorm2d(256)
    
    self.dropout1 = torch.nn.Dropout(p=0.1)
    self.dropout2 = torch.nn.Dropout(p=0.2)
    self.dropout3 = torch.nn.Dropout(p=0.4)
    
    self.fc_input_size = 256 * 18 * 18
    self.fc1 = torch.nn.Linear(self.fc_input_size, 256)
    self.fc2 = torch.nn.Linear(256, 120)
    self.fc3 = torch.nn.Linear(120, num_classes)

  def forward(self, x):

    x = self.pool(torch.relu(self.batch_norm1(self.conv1(x))))
    #x = self.dropout1(x)
    x = self.pool(torch.relu(self.batch_norm2(self.conv2(x))))
    #x = self.dropout2(x)
    x = self.pool(torch.relu(self.batch_norm3(self.conv3(x))))
    x = self.dropout3(x)
    
    x = x.view(-1, self.fc_input_size)
    
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    
    return x