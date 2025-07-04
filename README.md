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

📂 Project Structure
graphql
Copy
Edit
BrainTumorClassification/
├── main.py                         # Unified pipeline entry point
├── cnn_model.py                    # CNN architecture and classification logic
├── ann_model.py                    # ANN for glioma stage prediction
├── utils/                          # Helper functions for data loading/preprocessing
├── dataset/                        # Sample data if added
├── models/
│   ├── BTD_model.pth               # CNN model weights
│   ├── glioma_stages.pth           # ANN model weights
│   ├── BrainTumorClassification.ipynb   # Notebook for CNN testing
│   └── Glioma_Stages.ipynb               # Notebook for ANN training/testing
├── requirements.txt
├── README.md
└── .gitignore
🔍 Steps to Use
Upload a grayscale brain MRI image.

The CNN model will classify the tumor type.

If the prediction is "Glioma", enter the gene mutation test values.

The ANN model will return the predicted Glioma stage (I–IV).

✨ Features
✅ End-to-end deep learning pipeline

✅ Accurate multi-class tumor classification

✅ Secondary glioma staging system using ANN

✅ Lightweight and fast inference

✅ Clean, modular codebase (easy to modify)

✅ Fully implemented based on real medical research

🔮 Future Enhancements
💬 Integrate Gemini AI or a chatbot assistant for medical support

🧠 Apply transfer learning with models like ResNet, VGG, etc.

🌐 Deploy using Streamlit, Flask, or FastAPI

🖼️ Add support for DICOM image format used in clinical settings

👩‍💻 Author
Name: Afnan Shoukat

📧 Email: afnanshoukat011@gmail.com

🔗 LinkedIn: linkedin.com/in/afnan-shoukat-030306267

📝 License
This project is developed for academic and research purposes only.

You are welcome to fork, reuse, and reference it.

Please remember to give proper credit 🙏
