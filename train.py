import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_model
from tqdm import tqdm
import os
from sklearn.metrics import f1_score


def train(num_epochs=6, batch_size=16, learning_rate=0.0002):
    if not torch.cuda.is_available():
        print("CRITICAL WARNING: GPU (CUDA) is NOT available to PyTorch.")
        print("Please ensure you have an NVIDIA GPU and installed the correct PyTorch CUDA version.")
        print("Falling back to CPU, but training will be slow...")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    # 1. Get Data
    train_loader, val_loader, class_names = get_dataloaders(batch_size=batch_size)
    num_classes = len(class_names)
    print(f"Classes found: {class_names}")

    # 2. Get Model
    model = get_model(num_classes).to(device)

    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_macro_f1 = 0.0


    # 4. Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct_val / total_val

        # Calculate Macro F1
        val_macro_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Macro F1: {val_macro_f1:.4f}")

        # Save Best Model based on highest Macro F1
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"=> Saved new best model! (Best Val Macro F1: {best_macro_f1:.4f})")


    print("\nTraining complete! Best model saved as 'best_model.pth'")

if __name__ == '__main__':
    # Run training for 6 epochs
    train(num_epochs=6)
