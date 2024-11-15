import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as F


class SmileNonSmileDataset(Dataset):
    def __init__(self, data_dir, mode='train', augment=True, duplication_factor=5):
        self.data_dir = data_dir
        self.mode = mode  # 'train', 'val', or 'test'
        self.duplication_factor = duplication_factor

        self.base_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        self.augmentations = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(30,),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),),
            transforms.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation=3)
            ]

        self.image_paths = []
        self.labels = []
        smile_folder = os.path.join(data_dir, 'smile')
        non_smile_folder = os.path.join(data_dir, 'non_smile')

        for label, folder in enumerate([smile_folder, non_smile_folder]):
            for filename in os.listdir(folder):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder, filename)
                    self.image_paths.append(image_path)
                    self.labels.append(label)


        train_paths, test_paths, train_labels, test_labels = train_test_split(
            self.image_paths, self.labels, test_size=0.2, random_state=42, stratify=self.labels)

        train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels)


        if self.mode == 'train':
            self.image_paths = train_paths
            self.labels = train_labels
        elif self.mode == 'val':
            self.image_paths = val_paths
            self.labels = val_labels
        else:  # 'test' mode
            self.image_paths = test_paths
            self.labels = test_labels

        if self.mode == 'train':
            self.image_paths = self.image_paths * duplication_factor
            self.labels = self.labels * duplication_factor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image
        image = Image.open(img_path).convert('RGB')

        if self.mode == 'train':

            duplicate_index = idx % self.duplication_factor  # Will be between 0 and duplication_factor-1

            if duplicate_index == 0:
                # If duplicate_index == 0, don't apply augmentations (original image)
                image = self.base_transform(image)
            else:
                # For other duplicates, apply a random augmentation
                random_transform = random.choice(self.augmentations)
                image = random_transform(image)
                image = self.base_transform(image)  # Apply base transformation after augmentation

        else:
            # For validation or test mode: Only apply base transformations (no augmentations)
            image = self.base_transform(image)

        return image, label
    
class SmileDetectionCNN(nn.Module):
    def __init__(self):
        super(SmileDetectionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=3, stride=1, padding=0)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(50 * 30 * 30, 512)  # 50 channels, 30x30 spatial size after pooling
                
        self.fc2 = nn.Linear(512, 2)  # 2 classes for classification
        
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



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the input and output channels differ, we need a 1x1 convolution to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        return torch.relu(out)

class SmileDetectionResNet(nn.Module):
    def __init__(self):
        super(SmileDetectionResNet, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(8192, 512)  # 2 classes for classification
        self.fc2 = nn.Linear(512, 2)  # 2 classes for classification
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)


    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x





