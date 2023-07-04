import torch
import pandas as pd
from skimage import io
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.transform import resize
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import time
import os
import numpy as np
import pickle

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_time = time.time()
#print(torch.cuda.is_available())


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#Carga de datos
train_data = pd.read_csv('ISIC_2019_Train_data_GroundTruth_New.csv')

image_ids_train = train_data['image'].tolist()

path_train = [f"/home/nmercado/data_proyecto/data_proyecto/ISIC_2019_Training_Input/{image_id}.jpg" for image_id in image_ids_train]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_id = self.data['image'].iloc[index]
        image_path = f"/home/nmercado/data_proyecto/data_proyecto/ISIC_2019_Training_Input/{image_id}.jpg"
        image = resize(io.imread(image_path), ((32, 32)))
        if self.transform:
            image = self.transform(image)
        label = self.data['final_label'].iloc[index]
        return image, label

train_dataset = CustomDataset(train_data, transform=transform)

batch_train = 10

train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True)

for images, labels in train_loader:
    # Imprimir las imágenes y las etiquetas
    for i in range(len(images)):
        image = images[i]
        tamaño = image.size()
        label = labels[i]
        print(f'Imagen {i+1}:')
        print(image)
        print(tamaño)
        print(f'Etiqueta {i+1}: {label}')
        print('---------------------------')