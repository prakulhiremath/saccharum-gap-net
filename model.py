import torch.nn as nn
from torchvision import models


def get_model(num_classes):
    """
    ResNet18 model for Sugarcane Disease Classification
    """

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    # Replace final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
