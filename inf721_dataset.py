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

from inf721_cleaning import cleaning


class BirdDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        img_path = format(self.dataframe.iloc[idx, 1])
        image = Image.open(img_path).convert("RGB")

        label = torch.LongTensor([self.dataframe.iloc[idx, 0]]).squeeze()

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])


def dataset():

    df_train, df_valid, df_test = cleaning()

    train_dataset = BirdDataset(dataframe=df_train, transform=transform)
    test_dataset = BirdDataset(dataframe=df_test, transform=transform)
    valid_dataset = BirdDataset(dataframe=df_valid, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, valid_loader