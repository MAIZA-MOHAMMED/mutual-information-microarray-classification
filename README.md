[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-brightgreen.svg)](https://xgboost.readthedocs.io/)

🧬 **Mutual Information Outperforms Competing Feature Selection Methods for High-Dimensional Microarray Data Classification**

A comprehensive framework for feature selection and classification of microarray gene expression data using mutual information-based methods.

👥 **Authors**: Mohammed MAIZA, Chahira CHERIF, Samira CHOURAQUI, Abdelmalik TALEB-AHMED

📄 **Paper**: [Link to paper when published]

---

## 📋 Abstract

This paper introduces a novel and efficient framework for cancer classification using microarray data by integrating advanced mutual information (MI) criteria with modern machine learning techniques. We propose a systematic combination of **Joint Mutual Information (JMI)-based feature selection** and **Neural Network (NN) classifiers**. Our results show that JMI effectively preserves feature interactions that are critical for discrimination, enabling NN models to achieve near-perfect accuracy across multiple cancer types. The proposed JMI-NN pipeline significantly outperforms conventional MI-based methods, including Mutual Information Maximization (MIM) and Max-Relevance Min-Redundancy (MRMR), as well as other classifiers such as Random Forest (RF), XGBoost (XGB), and Support Vector Machines (SVM). Overall, the framework provides a robust and computationally efficient solution for genomic biomarker discovery.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Advanced Feature Selection** | Implements three MI-based methods (MIM, JMI, MRMR) for high-dimensional gene selection |
| 🤖 **Multiple Classifiers** | Random Forest (RF), XGBoost (XGB), Neural Networks (NN), and Support Vector Machines (SVM) |
| 📊 **Nine Cancer Datasets** | Comprehensive evaluation across diverse microarray datasets from GEO |
| ⚡ **Computational Efficiency** | Reduces model training time by 40-60% and model size by 70-90% |
| 🧬 **Biological Validation** | Identifies well-established cancer biomarkers (TP53, MYC, CDKN1A, EGR1) |
| 📈 **State-of-the-Art Performance** | Achieves up to 0.98 accuracy on Lymphoma and 0.97 on SRBCT |
| 🔄 **Reproducible Pipeline** | End-to-end framework from feature selection to classification |

---

## 🏆 Performance Highlights

### Best Results Across Datasets

| Dataset | Best Method | Accuracy | Precision | Recall | F1-Score |
|---------|-------------|----------|-----------|--------|----------|
| Lymphoma | JMI + NN | **0.98** | 0.98 | 0.97 | 0.97 |
| SRBCT | JMI + NN | **0.97** | 0.97 | 0.97 | 0.97 |
| DLBCL | JMI + NN | **0.95** | 0.95 | 0.94 | 0.94 |
| Colon Cancer | JMI + NN | **0.93** | 0.93 | 0.92 | 0.92 |
| Leukemia | JMI + NN | **0.91** | 0.91 | 0.90 | 0.90 |

*JMI consistently outperforms MIM and MRMR across all datasets (p < 0.05)*

### Comparison of Feature Selection Methods

| Dataset | MIM + NN | MRMR + NN | **JMI + NN** | Improvement |
|---------|----------|-----------|--------------|-------------|
| SRBCT | 0.92 | 0.93 | **0.97** | **+4-5%** |
| Lymphoma | 0.94 | 0.95 | **0.98** | **+3-4%** |
| DLBCL | 0.90 | 0.91 | **0.95** | **+4-5%** |
| Leukemia | 0.87 | 0.88 | **0.91** | **+3-4%** |
| Lung Cancer | 0.85 | 0.86 | **0.90** | **+4-5%** |

---

## 🗂️ Datasets

Nine publicly available microarray datasets from Gene Expression Omnibus (GEO):

| Dataset | Genes | Training Samples | Testing Samples | Classes | GEO Accession |
|---------|-------|------------------|-----------------|---------|---------------|
| Leukemia | 7,129 | 38 | 34 | 2 | GSE13159 |
| Brain Cancer | 10,367 | 60 | 30 | 2 | GSE50161 |
| Colon Cancer | 2,000 | 42 | 20 | 2 | GSE44861 |
| SRBCT | 2,308 | 63 | 20 | 2 | GSE16930 |
| Prostate Tumor | 12,600 | 102 | — | 2 | GSE6919 |
| Lung Cancer | 12,533 | 144 | 59 | 2 | GSE10072 |
| Lymphoma | 4,026 | 60 | 36 | 2 | GSE6338 |
| 11_Tumors | 4,200 | 119 | 55 | 2 | GSE14961 |
| DLBCL | 3,812 | 42 | — | 2 | GSE905 |

---
📋 Overview
This repository implements and compares mutual information-based feature selection methods for microarray data classification:
MIM (Mutual Information Maximization)
JMI (Joint Mutual Information)
MRMR (Max-Relevance Min-Redundancy)
Applied with multiple classifiers:
Neural Networks (best performer)
XGBoost
Support Vector Machines
Random Forest

🚀 Quick Start Installation
```bash
# Clone repository
git clone https://github.com/MAIZA-MOHAMMED/mutual-information-microarray-classification.git
cd mutual-information-microarray-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download datasets
cd data
python download_datasets.py

