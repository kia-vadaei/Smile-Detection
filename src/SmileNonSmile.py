import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch.nn as nn
from PIL import Image


class SmileNonSmileDataset(Dataset):
    def __init__(self, data_dir,):
    
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  
            transforms.Grayscale(num_output_channels=1),  
            transforms.ToTensor(),  
            # transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        self.image_paths = []
        self.labels = []
        
        smile_folder = os.path.join(data_dir, 'smile')
        non_smile_folder = os.path.join(data_dir, 'non_smile')
        
        for label, folder in enumerate([smile_folder, non_smile_folder]):
            for filename in os.listdir(folder):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder, filename)
                    self.image_paths.append(image_path)
                    self.labels.append(label)  # 0 for smile, 1 for non_smile

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')  
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
class SmileDetectionCNN(nn.Module):
    def __init__(self):
        super(SmileDetectionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=3, stride=1, padding=0)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(50 * 30 * 30, 500)  # 50 channels, 30x30 spatial size after pooling
                
        self.fc2 = nn.Linear(500, 2)  # 2 classes for classification
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x








