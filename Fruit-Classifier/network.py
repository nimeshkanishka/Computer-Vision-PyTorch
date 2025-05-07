import torch.nn as nn

class ClassifierNetwork(nn.Module):
    def __init__(self, num_classes: int):
        super(ClassifierNetwork, self).__init__()

        if not isinstance(num_classes, int):
            raise TypeError("num_classes must be an integer.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # [3, 64, 64] -> [32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # [32, 64, 64] -> [32, 32, 32]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # [32, 32, 32] -> [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # [64, 32, 32] -> [64, 16, 16]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # [64, 16, 16] -> [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # [128, 16, 16] -> [128, 8, 8]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # [128, 8, 8] -> [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # [256, 8, 8] -> [256, 4, 4]
            nn.Flatten(), # [256, 4, 4] -> [256 * 4 * 4]
        )

        fc_layers = [
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        ]
        if num_classes == 2:
            fc_layers.extend([
                nn.Linear(512, 1),
                nn.Sigmoid()
            ])
        else:
            fc_layers.append(nn.Linear(512, num_classes))

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))