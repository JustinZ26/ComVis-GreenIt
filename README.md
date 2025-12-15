# 🗑️ Waste Classifier

This project is an image-based waste classification system built using traditional machine learning (SVM) with traditional feature extraction techniques.

---

## 🔧 Running the Gradio App

Before running the Gradio app, you need to either:

1. **Train the model yourself** using the provided dataset, or  
2. **Download the pre-trained model files** from Google Drive:

**Pre-trained Model Files:**  
https://drive.google.com/drive/folders/1_nEomzP1LT0Sz8gOyqymog9PUSeikJ7h?usp=sharing

After downloading, place the following files into the `model/` directory alongsize `labels.json`:

- `svm_model.joblib`
- `scaler.joblib`
- `pca_transformer.joblib`

---

## 🌿 GreenIt (Web Application)

We also created a web-based application for this model, you can check it on the link below

**GreenIt Repository:**  
https://github.com/Sonnn30/GreenIt

---

## 🎰Feature Extraction Pipeline

Each image is resized to **150 × 150 pixels** and processed using the following pipeline

- **Color Features**: HSV color histogram (H, S, V channels)
- **Texture Features**: Gray Level Co-occurrence Matrix (GLCM)
  - Contrast
  - Energy
  - Homogeneity
  - Correlation
- **Edge Features**: Canny edge detection (edge density)
- **Shape & Structure**: Histogram of Oriented Gradients (HOG)
- **Local Texture**: Local Binary Pattern (LBP)
