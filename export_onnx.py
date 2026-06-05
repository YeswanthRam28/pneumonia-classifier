import torch
from model import get_model
import os

def export_to_onnx(weights_path='best_model.pth', output_path='best_model.onnx', num_classes=2):
    print(f"Loading weights from {weights_path}...")
    
    # Initialize model
    model = get_model(num_classes)
    
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found!")
        return
        
    # Load weights
    # Map to CPU because ONNX export handles CPU tensors well and it's safer for universal export
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    
    # Create a dummy input matching the shape of a single image (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Exporting model to {output_path}...")
    # Export the model
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True,
        opset_version=14, # Modern opset
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("Export complete! You can now use this model in the browser.")

if __name__ == '__main__':
    # Pneumonia dataset has 2 classes
    export_to_onnx(num_classes=2)
