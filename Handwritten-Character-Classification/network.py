import torch.nn as nn

class ClassifierNetwork(nn.Module):
    def __init__(
            self,
            image_size: int,
            image_channels: int,
            num_classes: int
    ):
        super(ClassifierNetwork, self).__init__()

        final_img_size = image_size / 16
        if final_img_size % 1 != 0:
            raise ValueError(f"image_size must be divisible by 16 to be compatible with the network architecture.")
        final_img_size = int(final_img_size)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1), # [img_channels, img_size, img_size] -> [32, img_size, img_size]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # [32, img_size, img_size] -> [32, img_size / 2, img_size / 2]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # [32, img_size / 2, img_size / 2] -> [64, img_size / 2, img_size / 2]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # [64, img_size / 2, img_size / 2] -> [64, img_size / 4, img_size / 4]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # [64, img_size / 4, img_size / 4] -> [128, img_size / 4, img_size / 4]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # [128, img_size / 4, img_size / 4] -> [128, img_size / 8, img_size / 8]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # [128, img_size / 8, img_size / 8] -> [256, img_size / 8, img_size / 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # [256, img_size / 8, img_size / 8] -> [256, img_size / 16, img_size / 16]
            nn.Flatten(), # [256, img_size / 16, img_size / 16] -> [256 * (img_size / 16) * (img_size / 16)]
        )
        
        fc_layers = [
            nn.Linear(256 * final_img_size * final_img_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.25)
        ]
        if num_classes == 2:
            fc_layers.extend([
                nn.Linear(1024, 1),
                nn.Sigmoid()
            ])
        else:
            fc_layers.append(nn.Linear(1024, num_classes))

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))