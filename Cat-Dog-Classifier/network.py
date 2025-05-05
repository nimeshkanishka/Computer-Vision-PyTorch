import torch.nn as nn

class ClassifierNetwork(nn.Module):
    def __init__(self):
        super(ClassifierNetwork, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # [3, 64, 64] -> [32, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # [32, 64, 64] -> [32, 32, 32]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # [32, 32, 32] -> [64, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # [64, 32, 32] -> [64, 16, 16]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # [64, 16, 16] -> [128, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # [128, 16, 16] -> [128, 8, 8]
            nn.Flatten(), # [128, 8, 8] -> [128 * 8 * 8]
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))