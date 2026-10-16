"""
Chess Neural Network (ResNet Policy-Value Architecture)
Inspired by AlphaZero / Leela Chess Zero architectures.
Accepts an 8x8x18 board tensor representation.
Outputs:
  - Value: [-1.0, 1.0] expected game outcome from current player's perspective.
  - Policy: Logits over possible move indices.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessBrainNet(nn.Module):
    def __init__(self, in_channels=18, filters=128, num_res_blocks=6):
        super().__init__()
        self.input_conv = ConvBlock(in_channels, filters)
        self.res_blocks = nn.ModuleList([ResBlock(filters) for _ in range(num_res_blocks)])
        
        # Value Head (evaluates winning probability [-1, 1])
        self.val_conv = nn.Conv2d(filters, 32, kernel_size=1)
        self.val_bn = nn.BatchNorm2d(32)
        self.val_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.val_fc2 = nn.Linear(128, 1)

        # Policy Head (prior probabilities for 64x64 move matrix = 4096 possible from-to moves)
        self.pol_conv = nn.Conv2d(filters, 64, kernel_size=1)
        self.pol_bn = nn.BatchNorm2d(64)
        self.pol_fc = nn.Linear(64 * 8 * 8, 4096)

    def forward(self, x):
        # x shape: [batch, 18, 8, 8]
        out = self.input_conv(x)
        for block in self.res_blocks:
            out = block(out)

        # Value
        v = F.relu(self.val_bn(self.val_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.val_fc1(v))
        value = torch.tanh(self.val_fc2(v))

        # Policy
        p = F.relu(self.pol_bn(self.pol_conv(out)))
        p = p.view(p.size(0), -1)
        policy = self.pol_fc(p)

        return policy, value
