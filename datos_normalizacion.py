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
from PIL import Image


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
    transforms.Resize((32, 32))
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
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.data['final_label'].iloc[index]
        return image, label

train_dataset = CustomDataset(train_data, transform=transform)

batch_train = 17700

train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True)

# Calcula las estadísticas de las imágenes de entrenamiento
channel_means = []
channel_stds = []
       
for image, _ in train_loader:
    image = image.to(device)
    # Reorganizar el tensor de imágenes
    image = image.view(3, -1)
    # Calcular la media y desviación estándar por canal
    mean = image.mean(dim=1)
    std = image.std(dim=1)
    # Guardar las medias y desviaciones estándar en listas
    channel_means.append(mean)
    channel_stds.append(std)

# Calcular las medias y desviaciones estándar promedio
mean = torch.mean(torch.stack(channel_means), dim=0)
std = torch.mean(torch.stack(channel_stds), dim=0)

print("Mean:", mean)
print("Std:", std)

breakpoint()
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

