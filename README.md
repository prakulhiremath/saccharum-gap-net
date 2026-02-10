# 🌿 Saccharum-GAP-Net

Deep learning model for sugarcane leaf disease detection using transfer learning and Global Average Pooling.

---

## 📌 Overview

Saccharum-GAP-Net is built to classify sugarcane leaf diseases from RGB images captured in real farm conditions. The model uses pretrained convolutional neural network backbones combined with Global Average Pooling (GAP) to reduce parameter count and control overfitting on limited agricultural datasets.

The project focuses on:

* High classification accuracy
* Stable performance on variable image quality
* Lightweight architecture suitable for real-world deployment

---

## 🧠 Model Approach

**Core Design**

* Transfer Learning (ImageNet pretrained backbone)
* Global Average Pooling instead of large dense layers
* Softmax disease classification head

**Why GAP?**

* Fewer trainable parameters
* Lower overfitting risk
* Better spatial feature summarization

---

## 📊 Dataset

**Name:** Sugarcane Leaf Disease Dataset
**Published:** 20 August 2022
**Version:** 1
**DOI:** 10.17632/9424skmnrk.1

**Contributors:**

* Swapnil Daphal
* Sanjay Koli

### Dataset Description

Manually collected sugarcane leaf disease image dataset containing five classes:

* Healthy
* Mosaic
* Redrot
* Rust
* Yellow disease

**Key Properties**

* Total Images: 2569
* Format: RGB (.jpg)
* Capture Source: Smartphones with different camera specs
* Image Size: Not fixed (device dependent)
* Region: Maharashtra, India
* Balanced class distribution
* Good diversity in lighting and capture conditions

**Collection Method**
Images were captured using smartphones with different resolutions to maintain real-world diversity. The dataset focuses specifically on leaf diseases in sugarcane plants.

**Institution**
Savitribai Phule Pune University

**Category**
Plant Diseases

**Dataset Size**
~160 MB (RAR archive)

---

## 🗂 Project Structure

```
saccharum-gap-net/
│
├── data/
├── models/
├── training/
├── evaluation/
├── inference/
├── notebooks/
└── README.md
```

---

## ⚙️ Training Pipeline

1. Image normalization
2. Data augmentation (rotation, brightness, blur, noise simulation)
3. Transfer learning backbone initialization
4. GAP feature aggregation
5. Dense classification head
6. Cross entropy / weighted loss training

---

## 📈 Evaluation Focus

* Class-wise accuracy
* Confusion matrix
* Field condition robustness
* Cross-device image generalization

---

## 🚀 Use Cases

* Farm disease monitoring
* Mobile-based crop health screening
* Research in agricultural computer vision
* Edge deployment experiments

---

## 📦 Requirements (Typical)

* Python 3.9+
* PyTorch / TensorFlow
* OpenCV
* NumPy
* scikit-learn

---

## 📜 Citation (Dataset)

If you use the dataset, cite using the DOI:

```
Sugarcane Leaf Disease Dataset (2022)
DOI: 10.17632/9424skmnrk.1
```

---

## 🤝 Acknowledgment

Dataset contributors:
Swapnil Daphal
Sanjay Koli

Savitribai Phule Pune University

---

## 📬 Notes

This repository is intended for research, academic, and applied agricultural ML development.
