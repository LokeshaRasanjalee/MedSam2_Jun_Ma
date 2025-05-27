import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(Simple3DCNN, self).__init__()
        # Input shape: [B, 1, 5, 512, 512]
        self.conv1 = nn.Conv3d(in_channels, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))  # Preserve temporal depth

        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))  # Again, only spatial pooling

        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))  # Output shape: [B, 64, 1, 1, 1]

        self.fc1 = nn.Linear(32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 16, 5, 512, 512]
        x = self.pool1(x)                    # [B, 16, 5, 256, 256]

        x = F.relu(self.bn2(self.conv2(x)))  # [B, 32, 5, 256, 256]
        x = self.pool2(x)                    # [B, 32, 5, 128, 128]

        x = F.relu(self.bn3(self.conv3(x)))  # [B, 64, 5, 128, 128]
        x = self.pool3(x)                    # [B, 64, 1, 1, 1]

        x = x.view(x.size(0), -1)            # Flatten to [B, 64]
        x = F.relu(self.fc1(x))              # [B, 256]
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
