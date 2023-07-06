import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import time
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_time = time.time()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#Carga de datos
train_data = pd.read_csv('ISIC_2019_Train_data_GroundTruth_New.csv')
test_data = pd.read_csv('ISIC_2019_Test_data_GroundTruth_New.csv')
valid_data = pd.read_csv('ISIC_2019_Valid_data_GroundTruth_New.csv')

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Redimensionar las imágenes a 224x224 (tamaño requerido por ResNet)
    transforms.ToTensor(),
    transforms.Normalize((0.5558, 0.5982, 0.6149), (0.2433, 0.1914, 0.1902))  # Normalización de los valores de los píxeles
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
test_dataset = CustomDataset(test_data, transform=transform)
valid_dataset = CustomDataset(valid_data, transform=transform)

# ... Código para crear los data loaders ... AJUSTAR BATCH
batch_train = 1773
batch_test = 380
batch_valid = 380

train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_test, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_valid, shuffle=False)

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

#Entrenamiento 
def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    train_predictions = []
    train_labels = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
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

    t_loss = train_loss/len(train_loader)
    acc = 100 * correct_predictions/total_samples
    
    return train_predictions, train_labels, t_loss, acc


#Validación y test 
def evaluate(model, data_loader, criterion):
    model.eval()
    loss = 0.0
    correct_predictions = 0
    total_samples = 0
    predictions = []
    labels = []

    with torch.no_grad():
        for images, labels_batch in data_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels_batch)

            loss += batch_loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels_batch).sum().item()
            total_samples += labels_batch.size(0)

            predictions.extend(predicted.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())

    avg_loss = loss / len(data_loader)
    accuracy = 100 * correct_predictions / total_samples

    return predictions, labels, avg_loss, accuracy

 
num_epochs = 10
for epoch in range(num_epochs):
    #impresión train
    train_predictions, train_labels, t_loss, acc = train(model, train_loader, criterion, optimizer)
    train_precision = precision_score(train_labels, train_predictions, average=None)
    train_recall = recall_score(train_labels, train_predictions, average=None)
    train_f1_score = f1_score(train_labels, train_predictions, average=None)

    print(f'Época: {epoch:.4f}')
    print(f'Training Loss: {t_loss:.4f} | Training Accuracy: {acc:.2f}%')
    print(f'Training Precision: {train_precision}')
    print(f'Training Recall: {train_recall}')
    print(f'Training F1-Score: {train_f1_score}')
    print('---------------------------')

    #Validación
    valid_predictions, valid_labels, v_loss, v_acc = evaluate(model, valid_loader, criterion)  
    valid_precision = precision_score(valid_labels, valid_predictions, average=None)
    valid_recall = recall_score(valid_labels, valid_predictions, average=None)
    valid_f1_score = f1_score(valid_labels, valid_predictions, average=None)

    print('---------- Validación ----------')
    print(f'Validation Loss: {v_loss:.4f} | Validation Accuracy: {v_acc:.2f}%')
    print(f'Validation Precision: {valid_precision}')
    print(f'Validation Recall: {valid_recall}')
    print(f'Validation F1-Score: {valid_f1_score}')
    print('-------------------------------')
    
    # Evaluación en el conjunto de prueba
    test_predictions, test_labels, t_loss, t_acc = evaluate(model, test_loader, criterion)
    test_precision = precision_score(test_labels, test_predictions, average=None)
    test_recall = recall_score(test_labels, test_predictions, average=None)
    test_f1_score = f1_score(test_labels, test_predictions, average=None)

    print('---------- Prueba ----------')
    print(f'Test Loss: {t_loss:.4f} | Test Accuracy: {t_acc:.2f}%')
    print(f'Test Precision: {test_precision}')
    print(f'Test Recall: {test_recall}')
    print(f'Test F1-Score: {test_f1_score}')
    print('----------------------------')
    
end_time = time.time()

# Cálculo del tiempo transcurrido en horas
elapsed_time = (end_time - start_time) / 60
print(f"Tiempo transcurrido: {elapsed_time:.2f} minutos")

