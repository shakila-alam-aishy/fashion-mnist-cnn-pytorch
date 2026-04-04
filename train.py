from model import CNNModel
from dataset import get_dataloaders

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 🔹 Seed
torch.manual_seed(42)

# 🔹 Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 🔹 Data
train_loader, test_loader = get_dataloaders()

# 🔹 Model
model = CNNModel().to(device)

# 🔹 Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.5
)

# 🔹 Training setup
epochs = 18
losses = []
train_accs = []

# 🔥 TRAINING
for epoch in range(epochs):

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_features, batch_labels in train_loader:

        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    acc = correct / total

    losses.append(avg_loss)
    train_accs.append(acc)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    scheduler.step()

# 🔥 EVALUATION
all_preds = []
all_labels = []

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_features, batch_labels in test_loader:

        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(batch_features)
        _, predicted = torch.max(outputs, 1)

        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

print("Test Accuracy:", correct / total)

# 🔥 CONFUSION MATRIX
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 🔥 TRAINING PLOTS
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(losses)
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(train_accs)
plt.title("Accuracy")

#plt.show(block=False)

torch.save(model.state_dict(), "model.pth")
print("✅ Model saved as model.pth")