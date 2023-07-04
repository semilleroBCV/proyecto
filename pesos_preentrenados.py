import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Especificar el dispositivo a utilizar (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carga de datos y preprocesamiento
train_data = pd.read_csv('ISIC_2019_Train_data_GroundTruth_New.csv')
#test_data = pd.read_csv('ISIC_2019_Test_data_GroundTruth_New.csv')
#valid_data = pd.read_csv('ISIC_2019_Valid_data_GroundTruth_New.csv')

# ... Código para cargar las rutas de las imágenes y crear los conjuntos de datos ...

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar las imágenes a 224x224 (tamaño requerido por ResNet)
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalización de los valores de los píxeles
])

train_dataset = CustomDataset(train_data, transform=transform)
#test_dataset = CustomDataset(test_data, transform=transform)
#valid_dataset = CustomDataset(valid_data, transform=transform)

# ... Código para crear los data loaders ...
batch_train = 10

train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True)


# Cargar el modelo pre-entrenado ResNet
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

# Reemplazar la capa completamente conectada para ajustarse al número de clases en tu problema
model.fc = nn.Linear(num_ftrs, 9)

model = model.to(device)

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

    train_loss /= total_samples
    accuracy = accuracy_score(train_labels, train_predictions)

    return accuracy, train_predictions, train_labels, train_loss

num_epochs = 10
for epoch in range(num_epochs):
    train_accuracy, train_predictions, train_labels, train_loss = train(model, train_loader, criterion, optimizer)

    train_precision = precision_score(train_labels, train_predictions, average=None)
    train_recall = recall_score(train_labels, train_predictions, average=None)
    train_f1_score = f1_score(train_labels, train_predictions, average=None)

    print(f'Training Loss: {train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}%')
    print(f'Training Precision: {train_precision}')
    print(f'Training Recall: {train_recall}')
    print(f'Training F1-Score: {train_f1_score}')
    print('---------------------------')

# ... Código adicional para evaluar el modelo en el conjunto de validación o prueba ...

