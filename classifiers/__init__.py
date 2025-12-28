"""
Classifier implementations for microarray data analysis.

This module provides optimized implementations of four classifiers:
1. Neural Networks (NN)
2. XGBoost (XGB)
3. Support Vector Machines (SVM)
4. Random Forest (RF)

All classifiers are optimized with hyperparameters as specified in the paper.
"""

from .neural_network import MicroarrayNeuralNetwork
from .xgboost_model import XGBoostClassifier
from .svm_model import SVMClassifier
from .random_forest import RandomForestClassifier

__version__ = "1.0.0"
__author__ = "Based on Cherif et al., 2022"

__all__ = [
    "MicroarrayNeuralNetwork",
    "XGBoostClassifier", 
    "SVMClassifier",
    "RandomForestClassifier"
]
