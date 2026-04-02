import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            # 🔹 Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 28 → 14

            # 🔹 Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2)  # 14 → 7
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)  # reshape
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x