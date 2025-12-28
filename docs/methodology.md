# Methodology : Mutual Information Feature Selection for Microarray Classification

## Overview

This document describes the comprehensive methodology for mutual information-based feature selection and classification of high-dimensional microarray data, as implemented in our research paper and codebase.

## 1. Problem Definition

### 1.1 Microarray Data Characteristics

Microarray data presents unique challenges for machine learning:

- **High Dimensionality**: Features (genes) >> Samples (patients)
  - Typically 5,000-20,000 genes vs 50-200 samples
  - "Large p, small n" problem

- **Noise and Redundancy**: 
  - Technical noise from measurement errors
  - Biological redundancy among correlated genes
  - Missing values due to experimental issues

- **Class Imbalance**: 
  - Unequal distribution of cancer subtypes
  - Rare cancer types with few samples

### 1.2 Research Objectives

1. **Feature Selection**: Identify informative gene subsets that maximize classification accuracy while maintaining biological relevance
2. **Method Comparison**: Evaluate three mutual information-based feature selection methods (MIM, JMI, MRMR)
3. **Classifier Evaluation**: Compare four machine learning classifiers (RF, XGBoost, NN, SVM)
4. **Biological Validation**: Ensure selected genes have known cancer associations

## 2. Dataset Collection and Preprocessing

### 2.1 Datasets Used

We analyzed nine publicly available microarray datasets:

| Dataset | Genes | Samples | Cancer Type | Source |
|---------|-------|---------|-------------|--------|
| Leukemia | 7,129 | 72 | Acute Lymphoblastic Leukemia | Golub et al., 1999 |
| Brain Cancer | 10,367 | 90 | Glioblastoma | Pomeroy et al., 2002 |
| Colon Cancer | 2,000 | 62 | Colorectal Adenocarcinoma | Alon et al., 1999 |
| SRBCT | 2,308 | 83 | Small Round Blue Cell Tumors | Khan et al., 2001 |
| Prostate Tumor | 12,600 | 102 | Prostate Adenocarcinoma | Singh et al., 2002 |
| Lung Cancer | 12,533 | 203 | Lung Adenocarcinoma | Bhattacharjee et al., 2001 |
| Lymphoma | 4,026 | 96 | Diffuse Large B-Cell Lymphoma | Alizadeh et al., 2000 |
| 11 Tumors | 4,200 | 174 | Multiple Tumor Types | Su et al., 2001 |
| DLBCL | 3,812 | 42 | Diffuse Large B-Cell Lymphoma | Shipp et al., 2002 |

### 2.2 Preprocessing Pipeline

```python
# Complete preprocessing steps
1. Quality Control
   - Remove probes with >20% missing values
   - Exclude samples with poor quality flags

2. Missing Value Imputation
   - K-nearest neighbors imputation (k=10)
   - Row-wise imputation for gene expression

3. Normalization
   - Quantile normalization across samples
   - Z-score standardization per gene

4. Filtering
   - Remove constant/near-constant features (variance < 0.01)
   - Eliminate genes with low expression (mean < threshold)

5. Label Encoding
   - Binary encoding for cancer/normal

   - Multi-class encoding for cancer subtypes
