import torch
import pandas as pd
from skimage import io
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.transform import resize
import time

#Carga de datos
train_data = pd.read_csv('ISIC_2019_Train_data_GroundTruth_New.csv')
test_data = pd.read_csv('ISIC_2019_Test_data_GroundTruth_New.csv')
valid_data = pd.read_csv('ISIC_2019_Valid_data_GroundTruth_New.csv')


image_ids_train = train_data['image'].tolist()
image_ids_test = test_data['image'].tolist()
image_ids_valid = valid_data['image'].tolist()


path_train = [f"/home/nmercado/data_proyecto/data_proyecto/ISIC_2019_Training_Input/{image_id}.jpg" for image_id in image_ids_train]
path_test = [f"/home/nmercado/data_proyecto/data_proyecto/ISIC_2019_Training_Input/{image_id}.jpg" for image_id in image_ids_test]
path_valid = [f"/home/nmercado/data_proyecto/data_proyecto/ISIC_2019_Training_Input/{image_id}.jpg" for image_id in image_ids_valid]

target_size = (224, 224)  # Tamaño objetivo de las imágenes

train_images = []
test_images = []
valid_images = []

train_images = [resize(io.imread(image_path), target_size) for image_path in path_train]
test_images = [resize(io.imread(image_path), target_size) for image_path in path_test]
valid_images = [resize(io.imread(image_path), target_size) for image_path in path_valid]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, images, transform=None):
        self.data = data
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_id = self.data['image'].iloc[index]
        image_path = f"data_proyecto/data_proyecto/{image_id}.jpg"
        image = io.imread(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.data['final_label'].iloc[index]
        return image, label

train_dataset = CustomDataset(train_data, train_images, transform=transform)
test_dataset = CustomDataset(test_data, test_images, transform=transform)
valid_dataset = CustomDataset(valid_data, valid_images, transform=transform)

start_time = time.time()

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True)

end_time = time.time()

# Cálculo del tiempo transcurrido
elapsed_time = end_time - start_time
print(f"Tiempo transcurrido: {elapsed_time} segundos")

#Modelo 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        #Capa convolucional 1 
        self.conv1 = nn.Conv2d(3,32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        #Capa convolucional 2 
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        #Capa convolucional 3
        self.conv3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Capa completamente conectada
        self.fc = nn.Linear(128, 9)  # Ajusta el tamaño de salida
        
    def forward(self,x):
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

def train(model, loader, criterion, optimizer):
    model.train()
    train_loss = 0.0 
    correct_predictions = 0 
    total_samples = 0
    true_positives = 0
    true_negatives = 0 
    false_positives = 0 
    false_negatives = 0
    
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optmizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # Calcular estadísticas para accuracy, precision, recall y F1-score
        true_positives += ((predicted == 1) & (labels == 1)).sum().item()
        true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
        false_positives += ((predicted == 1) & (labels == 0)).sum().item()
        false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    train_loss = train_loss / total_samples
    accuracy = correct_predictions / total_samples
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return train_loss, accuracy, precision, recall, f1_score
        
        