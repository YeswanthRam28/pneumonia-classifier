import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_model
from tqdm import tqdm
import os

def train(num_epochs=5, batch_size=16, learning_rate=0.001):
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

    best_val_loss = float('inf')

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

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct_val / total_val

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("=> Saved new best model!")

    print("\nTraining complete! Best model saved as 'best_model.pth'")

if __name__ == '__main__':
    # Run a quick training for 5 epochs
    train(num_epochs=5)
