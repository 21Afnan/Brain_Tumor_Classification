# 🧠 Brain Tumor Classification & Glioma Stage Detection

A deep learning-based diagnostic system built from scratch, leveraging MRI brain scans and gene mutation data to detect tumor types and predict glioma stages using CNN and ANN architectures.

---

## 📌 Overview

This project presents a **two-stage intelligent system** for medical image analysis:

- **Stage 1:** Classify brain MRI scans into four categories using a CNN:
  - No Tumor
  - Meningioma
  - Pituitary
  - Glioma

- **Stage 2:** If **Glioma** is detected, use a separate ANN model to predict its **stage (I–IV)** based on gene mutation test inputs.

---

## 📚 Research Inspiration

This work is inspired by a peer-reviewed research paper focused on deep learning-based brain tumor diagnosis.
[
> 🔗 **(https://onlinelibrary.wiley.com/doi/full/10.1155/2022/1830010)**
> *(Note: The paper did not provide datasets or implementation. The entire system here, including data collection, modeling, and integration, was developed independently.)*

---

## 🗂️ Dataset

- 📦 **Source:**  Brain Tumor MRI Dataset (Kaggle)
- 📸 Grayscale `.jpg` MRI images
- 📁 Categories: `Glioma`, `Meningioma`, `Pituitary`, `No Tumor`

---

## 🧠 Model Architectures

<details>
<summary>🟦 CNN - Tumor Type Classification</summary>

- **Input:** Grayscale Brain MRI
- **Layers:** 3 Conv2D layers + ReLU + MaxPooling
- **Classifier:** Fully connected layers → Softmax output
- **Classes:** 4 (No Tumor, Glioma, Meningioma, Pituitary)
- **Framework:** PyTorch

</details>

<details>
<summary>🟩 ANN - Glioma Stage Detection</summary>

- **Input:** Gene mutation test results (numerical)
- **Layers:** 2–3 Dense layers with ReLU
- **Output:** Glioma stage (I–IV)
- **Use Case:** Only triggered if CNN predicts "Glioma"

</details>

---

## 💾 Pre-trained Models

> All models trained from scratch using PyTorch.

| Model | Purpose | File Path |
|-------|---------|-----------|
| CNN   | Brain Tumor Classification | `models/BTD_model.pth` |
| ANN   | Glioma Stage Detection     | `models/glioma_stages.pth` |

⚠️ Models are for **inference only**. For training scripts, contact the author.

---

## 📂 Project Structure

BrainTumorClassification/
├── main.py                         # Unified pipeline entry point
├── cnn_model.py                    # CNN architecture and logic
├── ann_model.py                    # ANN architecture for stage prediction
├── utils/                          # Helper functions
├── dataset/                        # Sample image/test input (optional)
├── models/
│   ├── BTD_model.pth               # CNN model weights
│   ├── glioma_stages.pth           # ANN model weights
│   ├── BrainTumorClassification.ipynb  # Notebook for CNN testing
│   └── Glioma_Stages.ipynb              # Notebook for ANN use
├── requirements.txt
├── README.md
└── .gitignore
📥 Model Download (External)
Due to GitHub’s 100MB limit, trained models are hosted on Google Drive.

🔗 Download CNN Model (BTD_model.pth)
📁 Place it in: models/BTD_model.pth

Optional Auto-download code:

python
Copy
Edit
import os
import urllib.request

model_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
model_path = "models/BTD_model.pth"

if not os.path.exists(model_path):
    os.makedirs("models", exist_ok=True)
    print("Downloading model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded.")
⚙️ Installation & Setup
bash
Copy
Edit
# 1. Clone the repository
git clone https://github.com/your-username/BrainTumorClassification.git
cd BrainTumorClassification

# 2. Install dependencies
pip install -r requirements.txt
🚀 How to Run
bash
Copy
Edit
python main.py
🧾 Steps:

Upload a grayscale brain MRI image.

CNN predicts tumor type.

If Glioma, enter gene mutation test data.

ANN predicts glioma stage.

✨ Features
✅ Full pipeline: Image-based classification + test-based stage prediction
✅ Lightweight and fast inference
✅ Clean, modular architecture
✅ Extendable to other tumor types
✅ Based on real research

🧠 Future Improvements
Add Chatbot integration (e.g., Gemini AI for medical explanation)

Experiment with transfer learning (e.g., ResNet, VGG)

Deploy as web or mobile app

Add support for DICOM images

👤 Author
Afnan Shoukat
📧 Email: afnanshoukat011@gmail.com
🔗 LinkedIn: linkedin.com/in/afnan-shoukat-030306267

📝 License
This project is open for educational and academic research. Please give appropriate credit when using this repository.
