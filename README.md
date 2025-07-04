# 🧠 Brain Tumor Classification & Glioma Stage Detection

A deep learning-based diagnostic system built from scratch, leveraging MRI brain scans and gene mutation data to detect tumor types and predict glioma stages using CNN and ANN architectures.

---

## 📌 Overview

This project presents a **two-stage intelligent system** for brain tumor analysis:

### 🔹 Stage 1: Tumor Classification (CNN)
- **Input:** Grayscale MRI brain scan
- **Output Classes:** 
  - No Tumor  
  - Meningioma  
  - Pituitary  
  - Glioma  

### 🔹 Stage 2: Glioma Stage Detection (ANN)
- **Only activated if Glioma is detected**
- **Input:** Gene mutation test results (numerical)
- **Output:** Glioma Stage (I–IV)

---

## 📚 Research Inspiration

This work is inspired by a peer-reviewed research paper on brain tumor classification:

🔗 [Read the Research Paper](https://onlinelibrary.wiley.com/doi/full/10.1155/2022/1830010)

> 📝 *Note: The original research paper did not provide datasets or implementation. The entire pipeline—data collection, preprocessing, model development, and integration—was developed independently from scratch.*

---

## 🗂️ Dataset

- 📦 **Source:** [Kaggle - Brain MRI Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- 📸 Format: Grayscale `.jpg` images
- 📁 Classes: Glioma, Meningioma, Pituitary, No Tumor

---

## 🧠 Model Architectures

### 🟦 CNN - Tumor Type Classification
- 3 Conv2D layers + ReLU + MaxPooling
- Flatten + Fully Connected layers
- Output: 4-Class Softmax Classifier

### 🟩 ANN - Glioma Stage Detection
- Dense layers with ReLU activation
- Input: Numerical gene mutation values
- Output: Predicted Glioma Stage (I–IV)

---

## 💾 Pre-trained Models

All models were trained from scratch using PyTorch.

| Model | Purpose                    | File Path               |
|-------|----------------------------|--------------------------|
| CNN   | Brain Tumor Classification | `models/BTD_model.pth`   |
| ANN   | Glioma Stage Detection     | `models/glioma_stages.pth` |

📥 **Model Files Download:**  
🔗 [Google Drive - Model Folder](https://drive.google.com/drive/folders/1OCmobHiuUzU2kSIliDUxS2eUKwwwyhyD?usp=sharing)

After downloading, place the files in the `models/` directory.

> ⚠️ These models are for **inference only**. For training code, please contact the author.

---

## 📂 Project Structure

<pre> 📁 BrainTumorClassification/ ├── main.py # Unified pipeline entry point ├── cnn_model.py # CNN architecture and classification logic ├── ann_model.py # ANN for glioma stage prediction ├── utils/ # Helper functions for data loading/preprocessing ├── dataset/ # Sample data if added ├── models/ │ ├── BTD_model.pth # CNN model weights │ ├── glioma_stages.pth # ANN model weights │ ├── BrainTumorClassification.ipynb # Notebook for CNN testing │ └── Glioma_Stages.ipynb # Notebook for ANN training/testing ├── requirements.txt ├── README.md └── .gitignore </pre>
🔍 Steps:
Upload a grayscale brain MRI image.

CNN model classifies tumor type.

If prediction is "Glioma", enter gene mutation test values.

ANN model returns predicted Glioma stage (I–IV).

✨ Features
✅ End-to-end deep learning pipeline

✅ Accurate multi-class tumor classification

✅ Secondary glioma staging system

✅ Lightweight and fast inference

✅ Clean and modular codebase

✅ Independently implemented using real research

🔮 Future Enhancements
Integrate Gemini AI or chatbot assistant for medical support

Explore transfer learning (e.g., ResNet, VGG variants)

Deploy as a Streamlit, Flask, or FastAPI web app

Add support for DICOM medical image format

👩‍💻 Author
Afnan Shoukat
📧 Email: afnanshoukat011@gmail.com
🔗 LinkedIn: linkedin.com/in/afnan-shoukat-030306267

📝 License
This project is intended for academic and educational use only.
You're welcome to fork or reference it—just please give proper credit 🙏
