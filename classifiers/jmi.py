"""
Joint Mutual Information (JMI) feature selection.

JMI selects features by considering their joint mutual information with the target,
capturing synergistic relationships between features.
"""

import numpy as np
from itertools import combinations
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
from .utils import estimate_mutual_information

class JointMutualInformationSelector:
    """
    Joint Mutual Information feature selector.
    
    JMI selects features that together provide maximum information about the target.
    It considers interactions between features using the criterion:
    
    J(f) = I(f; y) - 1/|S| Σ_{s in S} [I(f; s) - I(f; s|y)]
    
    where S is the set of already selected features.
    
    References:
    -----------
    Yang, H. H., & Moody, J. (1999). Data visualization and feature selection: 
    New algorithms for nongaussian data. Advances in neural information processing systems, 12.
    """
    
    def __init__(self, n_features_to_select=100, method='jmi', 
                 discrete_target=True, random_state=None, verbose=1):
        """
        Initialize JMI selector.
        
        Parameters:
        -----------
        n_features_to_select : int
            Number of features to select
        method : str
            Feature selection method: 'jmi', 'mim', or 'mrmr'
        discrete_target : bool
            Whether the target variable is discrete
        random_state : int or None
            Random seed for reproducibility
        verbose : int
            Verbosity level (0: silent, 1: progress bar, 2: detailed)
        """
        self.n_features_to_select = n_features_to_select
        self.method = method.lower()
        self.discrete_target = discrete_target
        self.random_state = random_state
        self.verbose = verbose
        
        if self.method not in ['jmi', 'mim', 'mrmr']:
            raise ValueError("method must be 'jmi', 'mim', or 'mrmr'")
        
        self.selected_features = []
        self.feature_scores = {}
        self.mi_with_target = None
        self.mi_matrix = None
        
    def fit(self, X, y):
        """
        Select features using JMI criterion.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        n_samples, n_features = X.shape
        
        if self.n_features_to_select > n_features:
            raise ValueError(f"Cannot select {self.n_features_to_select} "
                           f"features from {n_features} available")
        
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Calculate mutual information with target for all features
        if self.verbose >= 1:
            print(f"Calculating mutual information with target...")
        
        self.mi_with_target = mutual_info_classif(
            X, y, 
            discrete_features=False,
            random_state=self.random_state,
            n_neighbors=3
        )
        
        if self.method == 'mim':
            # Simple MIM: select features with highest individual MI
            self.selected_features = np.argsort(self.mi_with_target)[-self.n_features_to_select:][::-1]
            return self
        
        # For JMI and MRMR, we need feature-feature mutual information
        if self.verbose >= 1:
            print(f"Calculating feature-feature mutual information...")
        
        # Precompute mutual information matrix (symmetric)
        self.mi_matrix = np.zeros((n_features, n_features))
        
        # Only compute upper triangle (including diagonal)
        for i in tqdm(range(n_features), desc="MI matrix", disable=self.verbose < 2):
            for j in range(i, n_features):
                if i == j:
                    self.mi_matrix[i, j] = self.mi_with_target[i]
                else:
                    mi = estimate_mutual_information(
                        X[:, i], X[:, j], 
                        discrete_x=False, 
                        discrete_y=False
                    )
                    self.mi_matrix[i, j] = mi
                    self.mi_matrix[j, i] = mi
        
        # Greedy forward selection
        self._greedy_forward_selection(X, y)
        
        return self
    
    def _greedy_forward_selection(self, X, y):
        """
        Perform greedy forward feature selection.
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            Target values
        """
        n_features = X.shape[1]
        selected = []  # S: set of selected features
        remaining = list(range(n_features))  # F: set of remaining features
        
        # Start with feature having maximum MI with target
        first_feature = np.argmax(self.mi_with_target)
        selected.append(first_feature)
        remaining.remove(first_feature)
        
        if self.verbose >= 1:
            print(f"\nStarting greedy forward selection ({self.method.upper()})")
            print(f"Initial feature: {first_feature} (MI={self.mi_with_target[first_feature]:.4f})")
        
        # Progress bar for selection
        pbar = tqdm(total=self.n_features_to_select - 1, 
                   desc=f"Selecting features ({self.method})",
                   disable=self.verbose < 1)
        
        while len(selected) < self.n_features_to_select and remaining:
            best_score = -np.inf
            best_feature = None
            
            for f in remaining:
                if self.method == 'jmi':
                    score = self._jmi_score(f, selected)
                elif self.method == 'mrmr':
                    score = self._mrmr_score(f, selected)
                else:
                    raise ValueError(f"Unknown method: {self.method}")
                
                if score > best_score:
                    best_score = score
                    best_feature = f
            
            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
                self.feature_scores[best_feature] = best_score
                
                if self.verbose >= 2:
                    print(f"Selected feature {best_feature} with score {best_score:.4f}")
                
                pbar.update(1)
            else:
                break
        
        pbar.close()
        
        self.selected_features = selected
        
        if self.verbose >= 1:
            print(f"\nSelected {len(selected)} features")
    
    def _jmi_score(self, candidate, selected):
        """
        Calculate JMI score for a candidate feature.
        
        JMI(f) = I(f; y) - 1/|S| Σ_{s in S} [I(f; s) - I(f; s|y)]
        
        For computational efficiency, we approximate I(f; s|y) ≈ 0
        when f and s are conditionally independent given y.
        
        Parameters:
        -----------
        candidate : int
            Candidate feature index
        selected : list
            List of already selected feature indices
            
        Returns:
        --------
        score : float
            JMI score
        """
        # Relevance term: I(f; y)
        relevance = self.mi_with_target[candidate]
        
        if not selected:
            return relevance
        
        # Redundancy term
        redundancy = 0
        for s in selected:
            # I(f; s) - I(f; s|y)
            mi_fs = self.mi_matrix[candidate, s]
            # Approximation: assume I(f; s|y) is small
            redundancy += mi_fs
        
        redundancy /= len(selected)
        
        # JMI score
        score = relevance - redundancy
        
        return score
    
    def _mrmr_score(self, candidate, selected):
        """
        Calculate MRMR score for a candidate feature.
        
        MRMR(f) = I(f; y) - 1/|S| Σ_{s in S} I(f; s)
        
        Parameters:
        -----------
        candidate : int
            Candidate feature index
        selected : list
            List of already selected feature indices
            
        Returns:
        --------
        score : float
            MRMR score
        """
        # Relevance term
        relevance = self.mi_with_target[candidate]
        
        if not selected:
            return relevance
        
        # Redundancy term
        redundancy = 0
        for s in selected:
            redundancy += self.mi_matrix[candidate, s]
        
        redundancy /= len(selected)
        
        # MRMR score
        score = relevance - redundancy
        
        return score
    
    def transform(self, X):
        """
        Reduce X to selected features.
        
        Parameters:
        -----------
        X : array-like
            Input data
            
        Returns:
        --------
        X_reduced : array-like
            Data with selected features
        """
        if not self.selected_features:
            raise ValueError("Must call fit before transform")
        
        return X[:, self.selected_features]
    
    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            Target values
            
        Returns:
        --------
        X_reduced : array-like
            Transformed data
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self, indices=True):
        """
        Get support mask or indices.
        
        Parameters:
        -----------
        indices : bool
            If True, return indices. If False, return boolean mask.
            
        Returns:
        --------
        support : array
            Indices or mask
        """
        if not self.selected_features:
            raise ValueError("Must call fit first")
        
        if indices:
            return np.array(self.selected_features)
        else:
            mask = np.zeros(self.mi_with_target.shape[0], dtype=bool)
            mask[self.selected_features] = True
            return mask
    
    def get_feature_importance(self):
        """
        Get feature importance.
        
        Returns:
        --------
        importance : dict
            Dictionary with importance information
        """
        if self.mi_with_target is None:
            raise ValueError("Must call fit first")
        
        return {
            'selected_features': self.selected_features.copy(),
            'mi_with_target': self.mi_with_target.copy(),
            'feature_scores': self.feature_scores.copy(),
            'method': self.method
        }
    
    def explain_selection(self, feature_names=None):
        """
        Explain feature selection results.
        
        Parameters:
        -----------
        feature_names : list or None
            Feature names
            
        Returns:
        --------
        explanation : str
            Text explanation
        """
        if not self.selected_features:
            return "No features selected yet."
        
        explanation = f"JMI Feature Selection Results ({self.method.upper()}):\n"
        explanation += "=" * 50 + "\n"
        explanation += f"Selected {len(self.selected_features)} features\n\n"
        
        explanation += "Top 10 selected features:\n"
        explanation += "-" * 30 + "\n"
        
        for i, feat_idx in enumerate(self.selected_features[:10], 1):
            mi_score = self.mi_with_target[feat_idx]
            feat_score = self.feature_scores.get(feat_idx, 0)
            
            if feature_names:
                feat_name = feature_names[feat_idx]
                explanation += f"{i:2d}. {feat_name} (idx={feat_idx})\n"
            else:
                explanation += f"{i:2d}. Feature {feat_idx}\n"
            
            explanation += f"    MI with target: {mi_score:.4f}\n"
            explanation += f"    Selection score: {feat_score:.4f}\n"
        
        return explanation


# Convenience function
def select_features_jmi(X, y, n_features=100, method='jmi', return_scores=False):
    """
    Quick JMI feature selection.
    
    Parameters:
    -----------
    X : array-like
        Input features
    y : array-like
        Target values
    n_features : int
        Number of features to select
    method : str
        'jmi' or 'mrmr'
    return_scores : bool
        Whether to return scores
        
    Returns:
    --------
    X_selected : array
        Selected features
    selected_indices : array
        Indices of selected features
    scores : dict, optional
        Scores if return_scores=True
    """
    selector = JointMutualInformationSelector(
        n_features_to_select=n_features,
        method=method,
        verbose=0
    )
    
    X_selected = selector.fit_transform(X, y)
    
    if return_scores:
        scores = selector.get_feature_importance()
        return X_selected, selector.selected_features, scores
    else:
        return X_selected, selector.selected_features