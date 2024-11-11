import torch.nn as nn
import torchvision.models as models
from net import CustomCNN
from torchvision.models import ResNet34_Weights, ResNet18_Weights

class MyModel(nn.Module):
    def __init__(self, backbone="resnet34", num_classes=3):
        super(MyModel, self).__init__()

        if backbone == "CustomCNN":
            self.models = CustomCNN()
        if backbone == "resnet18":
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.fc = nn.Linear(self.model.fc.in_features, 3)

        elif backbone == "resnet34":
            self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)   
            self.model.fc = nn.Linear(self.model.fc.in_features, 3)
        else:
            self.model = models.resnet50(num_classes=3)

    def forward(self, x):
        return self.model(x)