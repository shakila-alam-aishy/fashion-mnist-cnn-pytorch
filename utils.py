import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def evaluate_model(model, test_loader, device):
    """Evaluate model on test data and return accuracy and predictions"""

    model.eval()

    all_preds = []
    all_labels = []

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

    accuracy = correct / total
    return accuracy, all_labels, all_preds


def plot_confusion_matrix(all_labels, all_preds, labels=None):
    """Plot confusion matrix"""

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_training_curves(losses, train_accs):
    """Plot training loss and accuracy"""

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(losses)
    plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(train_accs)
    plt.title("Accuracy")

    plt.tight_layout()
    plt.show()