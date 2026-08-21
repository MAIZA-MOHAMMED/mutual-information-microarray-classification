[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-brightgreen.svg)](https://xgboost.readthedocs.io/)

🧬 **Mutual Information Outperforms Competing Feature Selection Methods for High-Dimensional Microarray Data Classification**

A comprehensive framework for feature selection and classification of microarray gene expression data using mutual information-based methods.

👥 **Authors**: Mohammed MAIZA, Chahira CHERIF, Samira CHOURAQUI, Abdelmalik TALEB-AHMED

📄 **Paper**: *Mutual Information Outperforms Competing Feature Selection Methods for High-Dimensional Microarray Data Classification*

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
| ⚡ **Computational Efficiency** | Reduces model training time by 40–60% and model size by 70–90% |
| 🧬 **Biological Validation** | Identifies well-established cancer biomarkers (TP53, MYC, CDKN1A, EGR1) |
| 📈 **State-of-the-Art Performance** | Achieves up to 0.98 accuracy on Lymphoma and 0.97 on SRBCT |
| 🔄 **Reproducible Pipeline** | End-to-end framework from feature selection to classification |

---

## 🏆 Performance Highlights

### Best Results Across All Datasets

| Dataset | Best Method | Accuracy | Precision | Recall | F1-Score |
|---------|-------------|----------|-----------|--------|----------|
| Lymphoma | JMI + NN | **0.98** | 0.98 | 0.97 | 0.97 |
| SRBCT | JMI + NN | **0.97** | 0.97 | 0.97 | 0.97 |
| DLBCL | JMI + NN | **0.95** | 0.95 | 0.94 | 0.94 |
| Colon Cancer | JMI + NN | **0.93** | 0.93 | 0.92 | 0.92 |
| Leukemia | JMI + NN | **0.91** | 0.91 | 0.90 | 0.90 |
| Lung Cancer | JMI + NN | **0.90** | 0.90 | 0.89 | 0.89 |
| Brain Cancer | JMI + NN | **0.85** | 0.85 | 0.84 | 0.84 |
| Prostate Tumor | JMI + NN | **0.82** | 0.82 | 0.81 | 0.81 |
| 11 Tumors | JMI + SVM | **0.73** | 0.73 | 0.72 | 0.72 |

*JMI paired with NN achieved the highest accuracy on **7 out of 9 datasets**. JMI significantly outperformed MIM and MRMR (paired t-test, p < 0.05).*

### Baseline Classification Accuracy (Without Feature Selection)

| Dataset | RF | XGB | NN | SVM |
|---------|-----|-----|-----|-----|
| Leukemia | 0.78 | 0.81 | 0.83 | 0.82 |
| Brain Cancer | 0.70 | 0.76 | 0.78 | 0.75 |
| Colon Cancer | 0.80 | 0.83 | 0.85 | 0.83 |
| SRBCT | 0.80 | 0.83 | 0.85 | 0.84 |
| Prostate Tumor | 0.66 | 0.70 | 0.73 | 0.70 |
| Lung Cancer | 0.76 | 0.80 | 0.82 | 0.80 |
| Lymphoma | 0.83 | 0.88 | 0.90 | 0.88 |
| 11 Tumors | 0.61 | 0.66 | 0.68 | 0.65 |
| DLBCL | 0.81 | 0.84 | 0.86 | 0.83 |

### Comparison of Feature Selection Methods (NN Classifier)

| Dataset | MIM + NN | MRMR + NN | **JMI + NN** | Improvement vs MIM | Improvement vs MRMR |
|---------|----------|-----------|--------------|-------------------|---------------------|
| Lymphoma | 0.94 | 0.96 | **0.98** | +4% | +2% |
| SRBCT | 0.92 | 0.95 | **0.97** | +5% | +2% |
| DLBCL | 0.91 | 0.93 | **0.95** | +4% | +2% |
| Colon Cancer | 0.90 | 0.92 | **0.93** | +3% | +1% |
| Leukemia | 0.88 | 0.89 | **0.91** | +3% | +2% |
| Lung Cancer | 0.86 | 0.89 | **0.90** | +4% | +1% |
| Brain Cancer | 0.80 | 0.84 | **0.85** | +5% | +1% |
| Prostate Tumor | 0.77 | 0.81 | **0.82** | +5% | +1% |
| 11 Tumors | 0.70 | 0.72 | **0.73** | +3% | +1% |

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
| 11 Tumors | 4,200 | 119 | 55 | 2 | GSE14961 |
| DLBCL | 3,812 | 42 | — | 2 | GSE905 |

All datasets were preprocessed following standard microarray analysis protocols: RMA normalization, log2 transformation, and missing value imputation using k-nearest neighbors.

---

## 📋 Overview

This repository implements and compares three mutual information-based feature selection methods for microarray data classification:

- **MIM** (Mutual Information Maximization)
- **JMI** (Joint Mutual Information)
- **MRMR** (Max-Relevance Min-Redundancy)

Applied with four state-of-the-art classifiers:
- **Neural Networks (NN)** — best performer
- **XGBoost (XGB)**
- **Support Vector Machines (SVM)**
- **Random Forest (RF)**

### Proposed Framework

1. **Feature Selection**: MI-based methods (MIM, JMI, MRMR) select the most informative gene subsets
2. **Classification**: ML algorithms (RF, XGB, NN, SVM) classify cancer types using selected features
3. **Evaluation**: Performance assessed via accuracy, precision, recall, and F1-score

Hyperparameters are optimized via grid search with 5-fold cross-validation. Training-test split follows an 80%–20% ratio.

---

## 🚀 Quick Start

### Installation

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
```

### Usage

```bash
# Run the complete pipeline
python main.py --dataset all --fs_method jmi --classifier nn

# Run for a specific dataset
python main.py --dataset lymphoma --fs_method jmi --classifier nn

# Compare all feature selection methods
python main.py --dataset all --fs_method all --classifier all
```

---

## 📁 Repository Structure

```
mutual-information-microarray-classification/
├── data/                   # Dataset download and preprocessing scripts
├── src/
│   ├── feature_selection/  # MIM, JMI, MRMR implementations
│   ├── classifiers/        # RF, XGB, NN, SVM models
│   ├── evaluation/         # Performance metrics and visualization
│   └── utils/              # Helper functions
├── results/                # Output results and figures
├── main.py                 # Main pipeline script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📊 Key Findings

- **JMI consistently outperforms MIM and MRMR** across all datasets when paired with NN
- The **JMI + NN combination** achieves the highest accuracy on 7 out of 9 datasets
- Feature selection reduces model training time by **40–60%**, decreases model size by **70–90%**, and improves inference speed by **30–50%**
- Selected feature sets include well-established cancer biomarkers: **TP53, MYC, CDKN1A, EGR1**
- The **11_Tumors dataset** proved most challenging, where JMI + SVM achieved the best result (0.73)

---

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@article{maiza2025mutual,
  title={Mutual Information Outperforms Competing Feature Selection Methods for High-Dimensional Microarray Data Classification},
  author={Maiza, Mohammed and Cherif, Chahira and Chouraqui, Samira and Taleb-Ahmed, Abdelmalik},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Paper**: [Link to paper when published]
- **GitHub**: https://github.com/MAIZA-MOHAMMED/mutual-information-microarray-classification
- **Datasets**: [Gene Expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/)

---

## 🙏 Acknowledgments

- Datasets sourced from NCBI Gene Expression Omnibus (GEO)
- This research received no specific grant from any funding agency
