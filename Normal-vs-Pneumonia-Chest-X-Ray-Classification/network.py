import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    """
    Standard residual block with optional downsampling.

    Attributes:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by this block.
        stride (int): Step size for the first convolution (default 1 means no downsampling).
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1
    ):
        super().__init__()

        # If input and output dimensions (channels, H, W) differ, use 1 x 1 conv to match them
        self.down = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
            if (stride != 1) or (in_channels != out_channels) # downsample if needed
            else nn.Identity() # otherwise pass the input through unchanged
        )

        # First 3 x 3 convolution (Changes resolution if stride != 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second 3 x 3 convolution (Retains the same resolution)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Save (possibly downsampled) input for the skip connection
        identity = self.down(x)

        # First conv -> BatchNorm -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv -> BatchNorm
        out = self.bn2(self.conv2(out))

        # Add skip connection and apply ReLU
        return F.relu(out + identity)
    

class SEBlock(nn.Module):

    """
    Squeeze-and-Excitation block for channel-wise attention.

    Attributes:
        in_channels (int): Number of channels in the input feature-map.
        r (int): Reduction ratio (Bottleneck size for the “squeeze” -> “excitation” transform).
    """

    def __init__(
            self,
            in_channels: int,
            r: int = 16
    ):
        super().__init__()

        # Bottleneck fully connected layers (in_channels -> in_channels / r -> in_channels)
        self.fc1 = nn.Linear(in_channels, in_channels // r)
        self.fc2 = nn.Linear(in_channels // r, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size() # batch size, channels, H, W

        # Global average pooling (Squeeze): [b, c, H, W] -> [b, c, H * W] -> [b, c]
        s = x.view(b, c, -1).mean(dim=2)

        # Excitation: FC -> ReLU -> FC -> Sigmoid to get weights in (0, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1) # [b, c] -> [b, c, 1, 1]

        # Scale input feature-maps by these weights
        return x * s


class ClassifierNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Stem: Conv (7 x 7 kernel, stride = 2) -> ReLU -> BatchNorm -> MaxPool (2 x 2 kernel)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
        )

        # First residual block + SE
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.se1 = SEBlock(64)

        # Second residual block + SE
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.se2 = SEBlock(128)

        # Dilated convolution (3 x 3 kernel, dilation = 2) to enlarge receptive field
        self.dilated = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)

        # Global pooling to 1 x 1 spatial size and flatten
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        # Dropout layers
        self.drop2d = nn.Dropout2d(p=0.1)
        self.drop = nn.Dropout(p=0.2)

        # Final linear layer to single output
        # No activation here as we are using binary_cross_entropy_with_logits loss function
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # Apply stem
        x = self.stem(x)

        # Residual + SE stage 1
        x = self.layer1(x)
        x = self.se1(x)
        x = self.drop2d(x)

        # Residual + SE stage 2
        x = self.layer2(x)
        x = self.se2(x)
        x = self.drop2d(x)

        # Dilated conv + ReLU
        x = F.relu(self.dilated(x))
        x = self.drop2d(x)

        # Pool to [b, 128, 1, 1] -> Flatten to [b, 128]
        x = self.pool(x).flatten(start_dim=1)
        x = self.drop(x)
        
        # Final linear layer -> [b, 1]
        x = self.fc(x)
        
        return x