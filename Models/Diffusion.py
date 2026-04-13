import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class DiffusionNetUNetStyle(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.channel_up1 = nn.Sequential(
            nn.Conv2d(in_channels + 1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.channel_up2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.channel_up3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res4 = ResidualBlock(64)

        self.channel_down4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res5 = ResidualBlock(32)
        # Atención descartada, eliminada para evitar OOM
        self.res6 = ResidualBlock(32)

        self.downsample3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.channel_down1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.channel_down2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.channel_down3 = nn.Sequential(
            nn.Conv2d(8, in_channels, kernel_size=3, padding=1),
            nn.Identity()
        )

    def forward(self, x, alpha):
        alpha_expanded = alpha[:, None, None, None].repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, alpha_expanded], dim=1)

        x = self.channel_up1(x)
        x = self.upsample1(x)
        x = self.channel_up2(x)
        x = self.upsample2(x)
        x = self.channel_up3(x)
        x = self.upsample3(x)
        x = self.res4(x)

        x = self.channel_down4(x)
        x = self.res5(x)
        # Atención eliminada
        x = self.res6(x)
        x = self.downsample3(x)
        x = self.channel_down1(x)
        x = self.downsample1(x)
        x = self.channel_down2(x)
        x = self.downsample2(x)
        x = self.channel_down3(x)
        return x
