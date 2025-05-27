import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Custom 3D CNN Model ----
class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=4, num_classes=4):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))  # downsample H, W

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((5, 1))  # keep T=5, reduce H=W=1

        self.fc = nn.Linear(64 * 5, num_classes)

    def forward(self, x):  # x: [B, 4, 1024, 1024]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 16, 1024, 1024]
        x = self.pool1(x)                    # [B, 16, 1024, 512]

        x = F.relu(self.bn2(self.conv2(x)))  # [B, 32, 1024, 512]
        x = self.pool2(x)                    # [B, 32, 1024, 256]

        x = F.relu(self.bn3(self.conv3(x)))  # [B, 64, 1024, 256]
        x = self.pool3(x)                    # [B, 64, 1024, 1]

        x = x.view(x.size(0), -1)            # [B, 64 * 5]
        x = self.fc(x)                       # [B, 4]
        return x
