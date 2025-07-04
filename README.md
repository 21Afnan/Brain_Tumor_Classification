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

## ⚠️ Inference-Only Notice

**These models are for inference only.**  
For training code, please contact the author.

---

## 📂 Project Structure

```bash
BrainTumorClassification/
├── main.py                     # Entry point: run this
├── ann_model.py                # ANN logic for glioma stage
├── cnn_model.py                # CNN architecture for image classification
├── utils/                      # Helper functions (preprocessing, loading, etc.)
├── dataset/                    # Image data (if any sample added)
├── models/
│   ├── BTD_model.pth           # (Download from Drive and place the file here)
│   ├── glioma_stages.pth           # Saved Model for Glioma Stages Detection
│   ├── BrainTumorClassification.ipynb   # Notebook for CNN testing
│   └── Glioma_Stages.ipynb               # Notebook for ANN training/testing
├── README.md
└── .gitignore
```

---


## 🔍 Steps to Use

- **Step 1:** Upload a grayscale brain MRI image.  
- **Step 2:** The CNN model classifies the tumor type.  
- **Step 3:** If the result is **Glioma**, enter gene mutation test values.  
- **Step 4:** The ANN model predicts the **Glioma stage (I–IV)**.

---

## ✨ Features

- ✅ **End-to-end deep learning pipeline**
- ✅ **Accurate multi-class tumor classification**
- ✅ **Glioma staging system using ANN**
- ✅ **Lightweight and fast inference**
- ✅ **Clean, modular codebase (easy to extend)**
- ✅ **Fully implemented based on real research**

---

## 🔮 Future Enhancements

- 💬 Integrate **Gemini AI** or chatbot assistant for medical explanations  
- 🧠 Apply **transfer learning** (e.g., ResNet, VGG) for higher accuracy  
- 🌐 Deploy as a **web app** using Streamlit, Flask, or FastAPI  
- 🖼️ Add support for **DICOM medical image format**

---

## 👩‍💻 Author

- **Name:** *Afnan Shoukat*  
- 📧 **Email:** [afnanshoukat011@gmail.com](mailto:afnanshoukat011@gmail.com)  
- 🔗 **LinkedIn:** [linkedin.com/in/afnan-shoukat-030306267](https://www.linkedin.com/in/afnan-shoukat-030306267)

---

## 📝 License

- This project is developed for **academic and research purposes only**.  
- You are welcome to **fork**, **reuse**, and **reference** it.  
- Please make sure to **give proper credit** 🙏

