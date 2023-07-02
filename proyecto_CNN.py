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


start_time = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

"""
print(len(path_train))
print(len(path_test))
print(len(path_valid))"""

target_size = (224, 224)  # Tamaño objetivo de las imágenes

train_images = [resize(io.imread(image_path), target_size) for image_path in path_train]
test_images = [resize(io.imread(image_path), target_size) for image_path in path_test]
valid_images = [resize(io.imread(image_path), target_size) for image_path in path_valid]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = device1
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


batch_train = 10
batch_test = 10
batch_valid = 10

train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_valid, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_test, shuffle=False)


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

        # Calcular estadísticas para accuracy, precision, recall y F1-score TERMINAR
        """
        true_positives += ((predicted == 1) & (labels == 1)).sum().item()
        true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
        false_positives += ((predicted == 1) & (labels == 0)).sum().item()
        false_negatives += ((predicted == 0) & (labels == 1)).sum().item()"""

    train_loss /= total_samples
    #accuracy = correct_predictions / total_samples
    accuracy = accuracy_score(train_labels, train_predictions)

    return accuracy, train_predictions, train_labels  

def validate(model, valid_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            val_predictions.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions / total_samples

    return accuracy, val_predictions, val_labels  

def evaluate(model, test_loader, criterion):
    model.eval()
    eval_loss = 0.0 
    correct_predictions = 0 
    total_samples = 0
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            eval_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

            
    accuracy = correct_predictions / total_samples
    """
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)"""

    #return eval_loss, accuracy, precision, recall, f1_score
    return accuracy, test_predictions, test_labels
    
num_epochs = 10
# for epoch in range(num_epochs):
#     train_accuracy = train(model,train_loader,criterion,optimizer)
#     test_accuracy = evaluate(model,train_loader,criterion,optimizer)
#     print(f'Training accuracy: {train_accuracy:.4f}')
#     print(f'Test accuracy: {test_accuracy:.4f}')
#     print('---------------------------')
for epoch in range(num_epochs):
    train_accuracy, train_predictions, train_labels = train(model, train_loader, criterion, optimizer)
    val_accuracy, val_predictions, val_labels = validate(model, valid_loader, criterion)
    test_accuracy, test_predictions, test_labels = evaluate(model, test_loader, criterion)

    train_precision = precision_score(train_labels, train_predictions, average=None)
    train_recall = recall_score(train_labels, train_predictions, average=None)
    train_f1_score = f1_score(train_labels, train_predictions, average=None)

    val_precision = precision_score(val_labels, val_predictions, average=None)
    val_recall = recall_score(val_labels, val_predictions, average=None)
    val_f1_score = f1_score(val_labels, val_predictions, average=None)

    test_precision = precision_score(test_labels, test_predictions, average=None)
    test_recall = recall_score(test_labels, test_predictions, average=None)
    test_f1_score = f1_score(test_labels, test_predictions, average=None)

    print(f'Training Loss: {train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}%')
    print(f'Training Precision: {train_precision}')
    print(f'Training Recall: {train_recall}')
    print(f'Training F1-Score: {train_f1_score}')
    print('---------------------------')

    print(f'Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%')
    print(f'Validation Precision: {val_precision}')
    print(f'Validation Recall: {val_recall}')
    print(f'Validation F1-Score: {val_f1_score}')
    print('---------------------------')

    print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test Precision: {test_precision}')
    print(f'Test Recall: {test_recall}')
    print(f'Test F1-Score: {test_f1_score}')
    print('---------------------------')


"""
for epoch in range(num_epochs):
    train_loss, train_accuracy, train_precision, train_recall, train_f1_score = train(model,train_loader,criterion,optimizer)
    test_loss, test_accuracy, test_precision, test_recall, test_f1_score = evaluate(model,train_loader,criterion,optimizer)


    print(f'Training Loss: {train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}%')
    print(f'Training precision: {train_precision:.4f} | Training Accuracy: {train_recall:.2f}%')
    print(f'F Score: {train_f1_score:.4f}')

    print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test precision: {test_precision:.4f} | Test Accuracy: {test_recall:.2f}%')
    print(f'F Score: {test_f1_score:.4f}')
    print('---------------------------') 
"""

end_time = time.time()

# Cálculo del tiempo transcurrido
elapsed_time = end_time - start_time
print(f"Tiempo transcurrido: {elapsed_time} segundos")



         