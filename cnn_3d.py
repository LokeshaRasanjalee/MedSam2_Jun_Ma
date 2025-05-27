import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=5, num_classes=4):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))  # [B, 64, 1, 1]

        self.fc1 = nn.Linear(64, 256)              # New FC layer
        self.fc2 = nn.Linear(256, num_classes)     # Final output layer

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 16, H, W]
        x = self.pool1(x)                    # Downsample

        x = F.relu(self.bn2(self.conv2(x)))  # [B, 32, H, W]
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))  # [B, 64, H, W]
        x = self.pool3(x)                    # [B, 64, 1, 1]

        x = x.view(x.size(0), -1)            # [B, 64]
        x = F.relu(self.fc1(x))              # [B, 128]
        x = self.fc2(x)                      # [B, num_classes]
        return x

    def count_parameters(self):
        print("Model Parameter Summary:")
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                total += param_count
                print(f"{name:<40} {param_count}")
        print(f"\nTotal Trainable Parameters: {total}")
