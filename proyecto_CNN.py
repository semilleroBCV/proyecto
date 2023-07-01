import torch
import pandas as pd
from skimage import io
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.transform import resize

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

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True)

