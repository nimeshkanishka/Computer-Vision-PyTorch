import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNet_B0(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=True)

        # Make all blocks of the features module of EfficientNet-B0 untrainable except for block 6 (last block)
        for name, param in self.backbone.features.named_parameters():
            if "6" not in name:
                param.requires_grad = False

        # Get the number of input features expected by the final classification layer of EfficientNet-B0
        in_features = self.backbone.classifier[1].in_features

        # Remove the classifier module of EfficientNet-B0 (We will use a custom classification module)
        self.backbone.classifier = nn.Identity()
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        # Custom classification module
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features // 2, 1),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ImageNet means and standard deviations (to normalize the images)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, -1, 1, 1)

    def forward(self, x):
        # Normalize each channel of the images to the ImageNet mean and standard deviation
        x = (x - self.mean) / self.std

        # [batch_size, 3, 224, 224] -> [batch_size, in_features]
        x = self.backbone(x)
        # [batch_size, in_features]
        x = self.dropout(x)
        
        # [batch_size, in_features] -> [batch_size, 1]
        x = self.classifier(x)

        return x