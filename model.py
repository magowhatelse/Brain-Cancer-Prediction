import torch.nn as nn
import torchvision.models as models
from net import CustomCNN
from torchvision.models import ResNet34_Weights, ResNet18_Weights

class MyModel(nn.Module):
    def __init__(self, backbone="vgg16", num_classes=3):
        super(MyModel, self).__init__()
        if backbone == "vgg16":
            self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 3)
            for param in self.model.classifier.parameters():
                param.requires_grad = False
            for param in self.model.classifier[6].parameters():
                param.requires_grad = True
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