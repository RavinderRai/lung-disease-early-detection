from torchvision import models
import torch.nn as nn

def create_model(num_classes=1):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model