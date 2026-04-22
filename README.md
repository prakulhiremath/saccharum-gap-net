# 🌿 Saccharum-GAP-Net  
### Lightweight Deep Learning for Sugarcane Leaf Disease Classification using Global Average Pooling

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19697946-green.svg)](https://doi.org/10.5281/zenodo.19697946)

---

## 📌 Overview

**Saccharum-GAP-Net** is a transfer learning framework for automatic classification of sugarcane leaf diseases from RGB images captured in real farm environments.

The central architectural idea is simple:

> Replace heavy fully connected layers with **Global Average Pooling (GAP)** to reduce parameter count, control overfitting, and enable lightweight deployment.

The framework is designed for real-world agricultural use, including mobile and edge-based screening tools.

---

## 🎯 Objectives

- Reduce overfitting on small agricultural datasets  
- Improve robustness to field lighting variations  
- Minimize model size without sacrificing accuracy  
- Support mobile and low-resource deployment  

---

## 🧠 Model Architecture

### Backbone (Transfer Learning)

Pretrained ImageNet models:
- ResNet  
- MobileNet  
- EfficientNet  

### Architectural Flow

```
Input Image (RGB)
        ↓
Pretrained CNN Backbone
        ↓
Feature Maps (C × H × W)
        ↓
Global Average Pooling
        ↓
1 × 1 × C Feature Vector
        ↓
Dense + Softmax
        ↓
Disease Class
```

### Why Global Average Pooling?

- Eliminates large fully connected layers  
- Significantly reduces parameters  
- Encourages spatial feature learning  
- Improves generalization  
- Supports Grad-CAM interpretability  

---

## 📊 Dataset

**Name:** Sugarcane Leaf Disease Dataset  
**Total Images:** 2,569  
**Image Type:** High-resolution RGB  
**Environment:** Unconstrained field conditions  
**Number of Classes:** 5  

### Classes

1. Healthy  
2. Mosaic  
3. Redrot  
4. Rust  
5. Yellow disease  

### Dataset Citation

Daphal, Swapnil; Koli, Sanjay (2022),  
“Sugarcane Leaf Disease Dataset”,  
Mendeley Data, V1,  
doi: 10.17632/9424skmnrk.1  

DOI Link: https://doi.org/10.17632/9424skmnrk.1

---

## 🗂 Repository Structure

```
saccharum-gap-net/
│
├── data/           # Data preprocessing & augmentation
├── models/         # GAP-based architecture definitions
├── training/       # Training loops & schedulers
├── evaluation/     # Metrics & confusion matrices
├── inference/      # Deployment-ready scripts
├── notebooks/      # Experiments & Grad-CAM
└── README.md
```

---

## 📈 Performance

| Metric | Score |
|--------|-------|
| Accuracy | 97.4% |
| Macro F1-Score | 0.96 |
| Inference Time | ~45 ms (mobile-tier CPU) |

Evaluation performed on stratified train–validation splits.

---

## 🔬 Training Details

- Loss Function: CrossEntropyLoss  
- Optimizer: Adam / SGD  
- Scheduler: StepLR / Cosine  
- Data Augmentation:
  - Random crop  
  - Horizontal flip  
  - Color jitter  
  - Lighting normalization  

Early stopping and weight decay are applied to control overfitting.

---

## 🔍 Explainability

Grad-CAM visualization is integrated to ensure:

- Model attention aligns with lesion regions  
- Background bias is reduced  
- Class-discriminative regions are meaningful  

---

## 📚 Citation (Methodology)

If you use this repository, please cite:

Hiremath, P., Galkar, A. (2026).  
**Saccharum-GAP-Net: Deep Learning for Sugarcane Disease Detection.**  
GitHub: https://github.com/prakulhiremath/saccharum-gap-net  

---

## ⚖️ License

MIT License  

Copyright (c) 2026 PRAKUL HIREMATH  

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction...

(Full MIT License applies.)

---

## 🤝 Acknowledgment

We acknowledge Swapnil Daphal and Sanjay Koli for publishing the Sugarcane Leaf Disease Dataset.

---
