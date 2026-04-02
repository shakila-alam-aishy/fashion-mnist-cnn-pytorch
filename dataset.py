import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index].reshape(1, 28, 28)
        y = self.labels[index]
        return x, y


def get_dataloaders(csv_path="fashion-mnist_test.csv", batch_size=32):

    # 🔹 Load data
    df = pd.read_csv(csv_path)

    # 🔹 Split features & labels
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # 🔹 Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔹 Normalization
    X_train = (X_train / 255.0 - 0.5) / 0.5
    X_test = (X_test / 255.0 - 0.5) / 0.5

    # 🔹 Dataset
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    # 🔹 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, test_loader