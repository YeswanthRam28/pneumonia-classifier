# Pneumonia Classifier CADx

An AI-based Computer-Aided Diagnosis (CADx) system for detecting Pneumonia from Chest X-Rays.

This project was built for a university project and uses a customized **EfficientNet** trained on the Medical MNIST (PneumoniaMNIST) dataset, reaching over **96% accuracy**.

The final model is deployed **100% Client-Side** using **ONNX Runtime Web**. This means the entire AI model runs directly in the user's browser securely, incredibly fast, and with absolutely zero backend server required!

## 🚀 Live Demo
Simply open `index.html` in your browser or visit the hosted GitHub Pages link. Upload an X-Ray image, and the AI will predict `Normal` or `Pneumonia` using your device's local compute power.

## 💻 Tech Stack
*   **Deep Learning:** PyTorch, Torchvision, TIMM (EfficientNet-B0)
*   **Dataset:** MedMNIST (PneumoniaMNIST)
*   **Deployment:** ONNX Runtime Web (JavaScript / WebAssembly)
*   **Frontend:** Vanilla HTML / CSS / JS

## 🛠️ Training the Model Locally

If you want to train the model yourself or modify the architecture:

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the PyTorch training script (Downloads the dataset and trains the model):
    ```bash
    python train.py
    ```

4.  Evaluate the model to see Confusion Matrix and Classification Report:
    ```bash
    python evaluate.py
    ```

5.  Export the `.pth` weights to `.onnx` for web deployment:
    ```bash
    python export_onnx.py
    ```
