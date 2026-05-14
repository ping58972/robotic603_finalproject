import torch
from torch import nn


class SymbolCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.30):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(32, 64),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(num_classes: int, dropout: float = 0.30) -> SymbolCNN:
    return SymbolCNN(num_classes=num_classes, dropout=dropout)

