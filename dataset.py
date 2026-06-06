import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import medmnist
from medmnist import INFO
import numpy as np

# We map 4 specific diseases from the 14-class dataset to indices 1-4.
# Index 0 is reserved for Normal.
TARGET_CLASSES = {
    1: 1, # Cardiomegaly
    2: 2, # Effusion
    4: 3, # Mass
    6: 4  # Pneumonia
}
CLASS_NAMES = ['Normal', 'Cardiomegaly', 'Effusion', 'Mass', 'Pneumonia']

class FiveClassMedMNIST(Dataset):
    def __init__(self, split, transform=None):
        info = INFO['chestmnist']
        DataClass = getattr(medmnist, info['python_class'])
        
        # Load the official dataset (downloads automatically if not found)
        self.medmnist_dataset = DataClass(split=split, transform=None, download=True)
        self.transform = transform
        
        # Fast vectorized filtering using the raw labels array
        labels_array = self.medmnist_dataset.labels
        sums = np.sum(labels_array, axis=1)
        
        temp_indices = []
        temp_labels = []
        
        print(f"Filtering {split} dataset for 5 classes...")
        for i in range(len(labels_array)):
            s = sums[i]
            if s == 0:
                # Patient is completely Normal
                temp_indices.append(i)
                temp_labels.append(0)
            elif s == 1:
                # Patient has exactly ONE disease
                disease_idx = int(np.argmax(labels_array[i]))
                if disease_idx in TARGET_CLASSES:
                    temp_indices.append(i)
                    temp_labels.append(TARGET_CLASSES[disease_idx])
                    
        if split == 'train':
            # Group indices by class
            class_groups = {0: [], 1: [], 2: [], 3: [], 4: []}
            for idx, label in zip(temp_indices, temp_labels):
                class_groups[label].append(idx)
                
            np.random.seed(42)
            self.valid_indices = []
            self.mapped_labels = []
            
            # Normal: Undersample to 4000
            normal_indices = np.random.choice(class_groups[0], size=min(4000, len(class_groups[0])), replace=False)
            self.valid_indices.extend(normal_indices)
            self.mapped_labels.extend([0] * len(normal_indices))
            
            # Others: Oversample to 2000 if less, or keep if more
            for label in [1, 2, 3, 4]:
                group = class_groups[label]
                target_size = 2000
                if len(group) < target_size:
                    # Oversample with replacement
                    oversampled = np.random.choice(group, size=target_size, replace=True)
                else:
                    # Keep as is
                    oversampled = group
                self.valid_indices.extend(oversampled)
                self.mapped_labels.extend([label] * len(oversampled))
                
            # Shuffle the final lists together
            combined = list(zip(self.valid_indices, self.mapped_labels))
            np.random.shuffle(combined)
            self.valid_indices, self.mapped_labels = zip(*combined)
            self.valid_indices = list(self.valid_indices)
            self.mapped_labels = list(self.mapped_labels)
        else:
            self.valid_indices = temp_indices
            self.mapped_labels = temp_labels
            
        print(f"Kept {len(self.valid_indices)} out of {len(labels_array)} images.")
        
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        mapped_label = self.mapped_labels[idx]
        
        # Get the PIL Image directly using the real index
        img = self.medmnist_dataset.imgs[real_idx]
        # imgs is a numpy array, we need to convert to PIL to apply torchvision transforms
        from PIL import Image
        img = Image.fromarray(img).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(mapped_label, dtype=torch.long)

def get_dataloaders(batch_size=16):
    print("Downloading MedMNIST Chest X-Ray dataset (Filtering for 5 Classes)...")
    
    # Standard ImageNet normalization values for EfficientNet with extra augmentation for oversampling
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = FiveClassMedMNIST(split='train', transform=train_transform)
    val_dataset = FiveClassMedMNIST(split='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, CLASS_NAMES

if __name__ == '__main__':
    train_loader, val_loader, classes = get_dataloaders()
    print(f"Classes: {classes}")
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        break
