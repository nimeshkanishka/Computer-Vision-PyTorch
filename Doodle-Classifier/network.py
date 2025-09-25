import torch.nn as nn

class ClassifierNetwork(nn.Module):
    def __init__(self):
        super(ClassifierNetwork, self).__init__()

        self.features = nn.Sequential(
            # Convolutional block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # [1, 28, 28] -> [32, 28, 28]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> [32, 14, 14]
            nn.Dropout2d(p=0.1),

            # Convolutional block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # -> [64, 14, 14]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> [64, 7, 7]
            nn.Dropout2d(p=0.1),

            # Convolutional block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # -> [128, 7, 7]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(4, 4)) # -> [128, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x