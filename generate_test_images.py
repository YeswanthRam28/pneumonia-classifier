import os
from PIL import Image
import numpy as np
from dataset import FiveClassMedMNIST

def generate_test_images():
    print("Loading validation dataset to extract sample images for each class...")
    val_dataset = FiveClassMedMNIST(split='val')
    
    # We want to find one image for each class: 0, 1, 2, 3, 4
    classes = ['Normal', 'Cardiomegaly', 'Effusion', 'Mass', 'Pneumonia']
    found = {}
    
    for idx in range(len(val_dataset)):
        real_idx = val_dataset.valid_indices[idx]
        label = val_dataset.mapped_labels[idx]
        
        if label not in found:
            # Get the image
            img_arr = val_dataset.medmnist_dataset.imgs[real_idx]
            # Convert to PIL Image
            img = Image.fromarray(img_arr).convert('RGB')
            # Resize to 224x224 so it's a nice standard size for the web dashboard upload
            img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
            
            class_name = classes[label]
            os.makedirs("sample_images", exist_ok=True)
            filename = os.path.join("sample_images", f"test_{class_name.lower()}.png")
            img_resized.save(filename)
            print(f"Saved {filename} (Class: {class_name}, Index: {label})")
            
            found[label] = filename
            
        if len(found) == 5:
            break
            
    print("\nAll test images generated successfully!")
    print("You can upload these files in the dashboard to test each class:")
    for label, filename in sorted(found.items()):
        print(f" - Class {label} ({classes[label]}): {filename}")

if __name__ == '__main__':
    generate_test_images()
