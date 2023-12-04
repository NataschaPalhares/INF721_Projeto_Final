import os
import sys
import torch
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix


def cleaning():

    csv_path = "birds.csv"
    df = pd.read_csv(csv_path)

    df = df.drop('scientific name', axis=1)
    df = df.drop('labels', axis=1)

    ###  No WINDOWS o path usa \ ao em vez de /
    ## df['filepaths'] = df['filepaths'].str.replace('/', '\\')   ### Desomentar esta linha caso voce use WINDOWS

    df_train = df[df['data set'] == 'train']
    df_train = df_train.drop('data set', axis=1)
    df_train['filepaths'] = df_train['filepaths'].str.replace('PARAKETT  AKULET', 'PARAKETT AUKLET') 

    df_test = df[df['data set'] == 'test']
    df_test = df_test.drop('data set', axis=1)
    df_test['filepaths'] = df_test['filepaths'].str.replace('PARAKETT  AKULET', 'PARAKETT  AUKLET')
    df_test = df_test.reset_index(drop=True)

    df_valid = df[df['data set'] == 'valid']
    df_valid = df_valid.drop('data set', axis=1)
    df_valid['filepaths'] = df_valid['filepaths'].str.replace('PARAKETT  AKULET', 'PARAKETT AUKLET')
    df_valid = df_valid.reset_index(drop=True)

    numSpecies = 11
    df_train = df_train[df_train['class id'].between(0, 0 + numSpecies)]
    df_valid = df_valid[df_valid['class id'].between(0, 0 + numSpecies)]
    df_test = df_test[df_test['class id'].between(0, 0 + numSpecies)]


    return df_train, df_valid, df_test
