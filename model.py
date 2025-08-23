import torch
import torch.nn as nn


# Residual block (ResNet-style)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Conv → BN → ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut path
        self.shortcut = nn.Sequential()

        # If shape changes (due to stride > 1 or channel mismatch),
        # use projection shortcut with 1x1 conv
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(x)
        out = out + shortcut
        out = torch.relu(out)
        return out


# Main CNN (ResNet-inspired)
class AudioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        # Stem conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # ResNet-style layers
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for _ in range(3)])
        self.layer2 = nn.ModuleList([ResidualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1) for i in range(4)])
        self.layer3 = nn.ModuleList([ResidualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1) for i in range(6)])
        self.layer4 = nn.ModuleList([ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1) for i in range(3)])

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feature_maps=False):
        feature_maps = {}

        # Stem
        x = self.conv1(x)
        feature_maps["conv1"] = x

        # Layers
        for block in self.layer1:
            x = block(x)
        feature_maps["layer1"] = x

        for block in self.layer2:
            x = block(x)
        feature_maps["layer2"] = x

        for block in self.layer3:
            x = block(x)
        feature_maps["layer3"] = x

        for block in self.layer4:
            x = block(x)
        feature_maps["layer4"] = x

        # Head
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        if return_feature_maps:
            return x, feature_maps
        return x
