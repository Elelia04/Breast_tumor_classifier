import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=2, fine_tune_layers=True):
    model = models.resnet50(pretrained=True)
    
    if fine_tune_layers:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze layer4 and fc
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model

