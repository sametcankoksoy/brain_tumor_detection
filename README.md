# 🧠 98% Accurate MRI-Based Tumor Classification System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Medical AI](https://img.shields.io/badge/Medical-AI-blue?style=for-the-badge)

This project implements a **Deep Learning** solution to detect brain tumors from MRI scans. By using **Transfer Learning** with the **ResNet18** architecture, the model can accurately distinguish between healthy brains and those with tumors.

## 🚀 Key Features
* **Framework:** Built with **PyTorch** for high performance.
* **Architecture:** Utilizes **ResNet18** (Pre-trained on ImageNet).
* **Reliability:** Provides **Confidence Scores** for every prediction.
* **Preprocessing:** Includes automated resizing, normalization, and skull-stripping logic.

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Language** | Python |
| **Deep Learning** | PyTorch, Torchvision |
| **Data Handling** | NumPy, Pandas |
| **Image Processing** | PIL (Pillow), OpenCV |

## 📊 Performance & Results
After training for 5 epochs, the model achieved an impressive **98% Accuracy** on the validation set.

* **High Sensitivity:** Minimal false negatives in tumor detection.
* **Reliability:** Proven performance on diverse MRI scan angles.

> **Example Result:**
> **Prediction:** Tumor Detected (Yes)  
> **Confidence:** 98.42%
