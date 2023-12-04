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

model.load_state_dict(torch.load('trained_model.pth'))

model.eval()

test_loss = 0.0
correct_predictions = 0
total_samples = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

average_test_loss = test_loss / len(test_loader)
accuracy = correct_predictions / total_samples

print(f"Test Loss: {average_test_loss}, Accuracy: {accuracy}")

conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()