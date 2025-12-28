"""
Feature selection module for microarray data analysis.

This module implements three mutual information-based feature selection methods:
1. MIM (Mutual Information Maximization)
2. JMI (Joint Mutual Information)
3. MRMR (Max-Relevance Min-Redundancy)

These methods are designed for high-dimensional data where features >> samples.
"""

from .mim import MutualInformationMaximization
from .jmi import JointMutualInformationSelector
from .mrmr import MaxRelevanceMinRedundancy
from .utils import (
    estimate_mutual_information,
    mutual_info_matrix,
    normalize_mi_scores,
    select_top_features
)

__version__ = "1.0.0"
__author__ = "Based on Cherif et al., 2022"

__all__ = [
    "MutualInformationMaximization",
    "JointMutualInformationSelector",
    "MaxRelevanceMinRedundancy",
    "estimate_mutual_information",
    "mutual_info_matrix",
    "normalize_mi_scores",
    "select_top_features"
]