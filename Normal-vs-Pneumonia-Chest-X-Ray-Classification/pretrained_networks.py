import torch.nn as nn
import torchvision.models as models


class EfficientNet_B0(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load EfficientNet-B0 with ImageNet weights
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last few blocks
        for param in self.backbone.features[6:].parameters():
            param.requires_grad = True
        
        # Get the number of input features expected by the final classification layer
        in_features = self.backbone.classifier[1].in_features

        # Remove the classifier module of EfficientNet-B0 (We will use a custom classification module)
        self.backbone.classifier = nn.Identity()
        
        # Custom classification module
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features // 2, 1),
        )

    def forward(self, x):
        # [batch_size, 3, 224, 224] -> [batch_size, in_features]
        x = self.backbone(x)
        
        # [batch_size, in_features] -> [batch_size, 1]
        x = self.classifier(x)

        return x
    

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load ResNet-50 with ImageNet weights
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze layer 4 (last conv block)
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # Get the size of the final fully connected layer
        in_features = self.backbone.fc.in_features

        # Remove the final fully connected layer (We will use a custom classification module)
        self.backbone.fc = nn.Identity()

        # Custom classification module
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features // 2, 1),
        )

    def forward(self, x):
        # [batch_size, 3, 224, 224] -> [batch_size, in_features]
        x = self.backbone(x)
        
        # [batch_size, in_features] -> [batch_size, 1]
        x = self.classifier(x)

        return x