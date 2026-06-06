import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
try:
    import seaborn as sns
except ImportError:
    sns = None

from dataset import get_dataloaders
from model import get_model

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load Data
    _, val_loader, class_names = get_dataloaders(batch_size=16)
    num_classes = len(class_names)

    # Load Model
    model = get_model(num_classes)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation on validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate Metrics
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    if sns is not None:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Lung Disease Classification')
        plt.savefig('confusion_matrix.png')
        print("\nSaved confusion matrix plot to 'confusion_matrix.png'")
    else:
        print("\nSeaborn not installed, skipping confusion matrix plot.")


if __name__ == '__main__':
    evaluate()
