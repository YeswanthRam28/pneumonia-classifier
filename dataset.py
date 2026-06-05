import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import medmnist
from medmnist import INFO
import numpy as np

class PneumoniaMedMNIST(Dataset):
    def __init__(self, split, transform=None):
        info = INFO['pneumoniamnist']
        DataClass = getattr(medmnist, info['python_class'])
        
        # Load the official dataset (downloads automatically if not found)
        self.medmnist_dataset = DataClass(split=split, transform=None, download=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.medmnist_dataset)
        
    def __getitem__(self, idx):
        # image is a PIL Image, label is a numpy array of shape (1,)
        img, label = self.medmnist_dataset[idx]
        
        # Ensure 3 channels for EfficientNet
        img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # Extract the single integer label
        binary_label = int(label[0])
            
        return img, torch.tensor(binary_label, dtype=torch.long)

def get_dataloaders(batch_size=16):
    print("Downloading MedMNIST Pneumonia dataset...")
    
    # Standard ImageNet normalization values for EfficientNet
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = PneumoniaMedMNIST(split='train', transform=train_transform)
    val_dataset = PneumoniaMedMNIST(split='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    info = INFO['pneumoniamnist']
    class_names = [info['label'][str(i)] for i in range(len(info['label']))]
    
    return train_loader, val_loader, class_names

if __name__ == '__main__':
    train_loader, val_loader, classes = get_dataloaders()
    print(f"Classes: {classes}")
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        break
