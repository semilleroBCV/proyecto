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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_time = time.time()
#print(torch.cuda.is_available())


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#Carga de datos
train_data = pd.read_csv('ISIC_2019_Train_data_GroundTruth_New.csv')
#test_data = pd.read_csv('ISIC_2019_Test_data_GroundTruth_New.csv')
#valid_data = pd.read_csv('ISIC_2019_Valid_data_GroundTruth_New.csv')


#image_ids_train = train_data['image'].tolist()
#image_ids_test = test_data['image'].tolist()
#image_ids_valid = valid_data['image'].tolist()


path_train = [f"/home/nmercado/data_proyecto/data_proyecto/ISIC_2019_Training_Input/{image_id}.jpg" for image_id in image_ids_train]
#path_test = [f"/home/nmercado/data_proyecto/data_proyecto/ISIC_2019_Test_Input/{image_id}.jpg" for image_id in image_ids_test]
#path_valid = [f"/home/nmercado/data_proyecto/data_proyecto/ISIC_2019_Valid_Input/{image_id}.jpg" for image_id in image_ids_valid]
#target_size = (782, 647)  # Tamaño objetivo de las imágenes


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar las imágenes a 224x224 (tamaño requerido por ResNet)
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalización de los valores de los píxeles
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
#test_dataset = CustomDataset(test_data, transform=transform)
#valid_dataset = CustomDataset(valid_data, transform=transform)

batch_train = 196
#batch_test = 10
#batch_valid = 10

train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True)
#valid_loader = DataLoader(valid_dataset, batch_size=batch_valid, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=batch_test, shuffle=False)

#print(train_loader)

#Modelo 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Capa convolucional 1: entrada 3 canales, salida 32 canales, kernel de 3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Capa convolucional 2: entrada 32 canales, salida 64 canales, kernel de 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Capa convolucional 3: entrada 64 canales, salida 128 canales, kernel de 3x3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Capa completamente conectada: entrada 128*4*4 unidades, salida 9 unidades (número de clases)
        self.fc = nn.Linear(128 * 4 * 4, 9)
    
    def forward(self, x):
        # Capa convolucional 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Capa convolucional 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Capa convolucional 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Aplanar los mapas de características
        x = x.view(x.size(0), -1)
        
        # Capa completamente conectada
        x = self.fc(x)
        
        return x

model = CNN().to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0 
    correct_predictions = 0 
    total_samples = 0
    train_predictions = []
    train_labels = []
    
    
    for images, labels in train_loader:
        images, labels = images.to(device).float(), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        train_predictions.extend(predicted.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_loss /= total_samples
    t_loss = train_loss/len(train_loader)
    acc = 100 * correct_predictions/total_samples
    accuracy = accuracy_score(train_labels, train_predictions)

    return acc, accuracy,train_predictions, train_labels, train_loss, t_loss

 
num_epochs = 10
for epoch in range(num_epochs):
    acc_manual, train_accuracy, train_predictions, train_labels, train_loss, t_loss_manual = train(model, train_loader, criterion, optimizer)

    train_precision = precision_score(train_labels, train_predictions, average=None)
    train_recall = recall_score(train_labels, train_predictions, average=None)
    train_f1_score = f1_score(train_labels, train_predictions, average=None)

    print(f'Training Loss: {train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}%')
    print(f'Training Loss manual: {t_loss_manual:.4f} | Training Accuracy manual: {acc_manual:.2f}%')
    print(f'Training Precision: {train_precision}')
    print(f'Training Recall: {train_recall}')
    print(f'Training F1-Score: {train_f1_score}')
    print('---------------------------')


end_time = time.time()

# Cálculo del tiempo transcurrido en horas
elapsed_time = (end_time - start_time) / 3600
print(f"Tiempo transcurrido: {elapsed_time:.2f} horas")

