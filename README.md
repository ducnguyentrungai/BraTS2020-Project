# 🧠 BraTS2020 Multitask: Tumor Segmentation + Survival Classification

Welcome to the official repository for a **multitask learning project** built on the **BraTS2020** dataset.  
This project aims to **simultaneously segment brain tumors** and **predict patient survival categories** using multi-modal MRI scans and clinical data.

> Developed with care by **Nguyen Trung Duc** ✨

![PyTorch](https://img.shields.io/badge/framework-PyTorch-red?style=flat&logo=pytorch)
![MONAI](https://img.shields.io/badge/medical-MONAI-orange?style=flat&logo=medical-services)
![Python](https://img.shields.io/badge/language-Python-blue?style=flat&logo=python)

---

## 📌 Overview

This multitask model performs:

- **3D Brain Tumor Segmentation** from MRI volumes (modalities: T1, T1ce, T2, FLAIR)
- **Survival Days Classification** into 3 categories:
  - Short-survivor: < 387 days
  - Mid-survivor: 388 -946 days
  - Long-survivor: > 946 days

Segmentation helps extract spatial tumor structure, while classification uses both **image features** and **tabular clinical data**.

---

## 🧠 Model Architecture

The model integrates:

- `SwinUNETR` backbone from MONAI for segmentation
- A **feature fusion module** (attention or concatenation)
- A **classification head** for survival prediction

