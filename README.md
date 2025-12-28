# 🧬 Mutual Information Microarray Classification

A comprehensive framework for feature selection and classification of microarray gene expression data using mutual information-based methods.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This repository implements and compares mutual information-based feature selection methods for microarray data classification:

- **MIM** (Mutual Information Maximization)
- **JMI** (Joint Mutual Information) 
- **MRMR** (Max-Relevance Min-Redundancy)

Applied with multiple classifiers:
- Neural Networks (best performer)
- XGBoost
- Support Vector Machines
- Random Forest

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
