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

from inf721_model import CNN
from inf721_dataset import dataset
from inf721_cleaning import cleaning


df_train, df_valid, df_test = cleaning()

train_loader, test_loader, valid_loader = dataset()


num_classes = len(df_train['class id'].unique())
model = CNN(num_classes=num_classes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 20

train_losses = []
valid_losses = []

for epoch in range(epochs):

    model.train()
    epoch_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    average_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(average_epoch_loss)

    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            valid_loss += loss.item()

    average_valid_loss = valid_loss / len(valid_loader)
    valid_losses.append(average_valid_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {average_epoch_loss}, Valid Loss: {average_valid_loss}")

torch.save(model.state_dict(), 'trained_model.pth')


plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Training Loss')
plt.plot(range(1, epochs + 1), valid_losses, marker='o', label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.legend()
plt.grid(True)
plt.show()


def getModel():
    
    return model