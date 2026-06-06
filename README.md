# PNEUMA | 5-Class Scan Analysis Dashboard & CADx

An advanced, end-to-end Computer-Aided Diagnosis (CADx) system for detecting and explaining chest diseases from X-Ray scans. 

This project integrates a client-side **ONNX Deep Learning model** for image classification with an **Nvidia NIM LLM chatbot** for generating patient communication and clinical insights.

---

## ✨ Features

1. **5-Class Neural Classifier:**
   Classifies chest X-Rays into 5 distinct categories:
   * **Normal** (No diagnostic findings)
   * **Cardiomegaly** (Enlargement of the heart)
   * **Effusion** (Fluid collection in pleural cavities)
   * **Mass** (Abnormal tissue growth/lump)
   * **Pneumonia** (Lung infection/inflammation)
2. **100% Client-Side Inference:**
   Uses **ONNX Runtime Web (WebAssembly)** to execute the trained **EfficientNet-B0** network directly in the user's web browser. Zero server-side inference overhead.
3. **PNEUMA AI Assistant:**
   Powered by Nvidia NIM (`minimaxai/minimax-m2.7`). As soon as a scan is classified, the assistant automatically drafts a clinical insight and opens an interactive chat panel where you can ask follow-up questions.
4. **Sleek Cyberpunk/Medical Aesthetics:**
   Beautiful, responsive design system featuring scanning grid animations, dynamic differential diagnosis charts, and a floating agent panel.

---

## 💻 Tech Stack
* **Deep Learning:** PyTorch, Torchvision (EfficientNet-B0)
* **Dataset:** MedMNIST v2 (ChestMNIST)
* **Web Inference:** ONNX Runtime Web (WASM)
* **LLM Engine:** Nvidia NIM API (`minimaxai/minimax-m2.7` model)
* **Web UI:** HTML5, CSS3, Tailwind CSS, JavaScript (ES6)

---

## 🛠️ Local Installation & Development

### 1. Set Up Environment
Create a Python virtual environment and install dependencies:
```bash
# Create and activate environment
python -m venv venv
.\venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt
```

### 2. Dataset & Training (Optional)
The project includes a balanced dataset configuration (`dataset.py`) that handles severe class imbalances via undersampling the Normal class and oversampling disease classes with active data augmentation (random rotation, translation, and color jitter).
```bash
# Train the model (automatically uses GPU if CUDA is available)
python train.py

# Export the trained model to ONNX format
python export_onnx.py
```

### 3. Run the Local Web App & CORS Proxy
To chat with the Nvidia LLM, the browser needs to fetch completion data. Direct calls to Nvidia NIM from `localhost` fail due to CORS. We run a lightweight local proxy:
```bash
python server.py
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 🧪 Testing the Classifier
We have provided sample X-Ray images extracted directly from the MedMNIST validation dataset inside the `sample_images/` directory:
* `sample_images/test_normal.png`
* `sample_images/test_cardiomegaly.png`
* `sample_images/test_effusion.png`
* `sample_images/test_mass.png`
* `sample_images/test_pneumonia.png`

Simply upload any of these files using the **Digital Input Zone** on the dashboard, click **Run Classification**, and verify the diagnostic output!

---

## 🌐 Deployment & Hosting Guide

### Option 1: GitHub Pages (Static Hosting)
Since the neural network runs entirely in the browser using WebAssembly, you can host this project 100% for free on **GitHub Pages**!

1. Go to your repository settings on GitHub.
2. Under **Pages**, select deployment from the `main` branch.
3. Once active, your web application will be live at `https://<username>.github.io/<repo-name>/`.

> [!NOTE]
> When hosted on GitHub Pages, the Nvidia NIM LLM Chatbot will encounter CORS errors because the API is called directly from the browser. To resolve this for public deployment, you can configure Nvidia API endpoints behind a serverless function (e.g., Vercel, Netlify) or use our Option 2.

### Option 2: Render / Heroku (Full Stack Hosting)
Deploy both the frontend and the `server.py` file to a hosting service like Render, Heroku, or railway.app.
1. Create a `Procfile` containing: `web: python server.py`
2. Set your environment variables and launch. The server will host static files and handle the API proxy requests automatically!
