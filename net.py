import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import torch.nn.functional as F


class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):  
        super(CustomCNN, self).__init__()
        
        # 1. conv block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. conv block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3. conv block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate size of flattened features
        # Input: 224x224 -> 112x112 -> 56x56 -> 28x28
        self.flatten_size = 256 * 28 * 28
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# class MiniAlexNet(nn.Module):
#     def __init__(self):
#         super(MiniAlexNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  
#         self.pool2 = nn.MaxPool2d(2, stride=2) 
        
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(16384, 4096) 
#         self.fc2 = nn.Linear(4096, 3)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # ----conv1 --------
#         out = self.conv1(x)
#         out = self.relu(out) 
#         out = self.pool1(out) 
#         # ---- conv2 --------
#         out = self.conv2(out)
#         out = self.relu(out) 
#         out = self.pool2(out)
#         # --- conv3 ----------
#         out = self.conv3(out)
#         out = self.relu(out) 
#         # ---- MLP ------------
#         out = self.flatten(out)
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out
