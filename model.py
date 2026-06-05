import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_model(num_classes):
    # Load pre-trained EfficientNet-B0 on ImageNet
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    
    # Replace the final classification head (classifier)
    # EfficientNet classifier is a Sequential block. We replace the Linear layer.
    # The in_features of the last layer in B0 is 1280.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == '__main__':
    model = get_model(num_classes=4)
    print("Model initialized successfully!")
    # Test a dummy forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}") # Should be [2, 4]
