"""
Max-Relevance Min-Redundancy (MRMR) feature selection.

MRMR selects features that have maximum relevance to the target 
while having minimum redundancy among themselves.
"""

import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
from .utils import estimate_mutual_information

class MaxRelevanceMinRedundancy:
    """
    Max-Relevance Min-Redundancy feature selector.
    
    MRMR aims to find a subset of features that jointly have the largest dependency
    on the target variable (max-relevance) and the smallest pairwise dependency
    among themselves (min-redundancy).
    
    The criterion is: max[I(f_i; y) - 1/|S| Σ_{f_j in S} I(f_i; f_j)]
    
    References:
    -----------
    Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual 
    information criteria of max-dependency, max-relevance, and min-redundancy. 
    IEEE Transactions on pattern analysis and machine intelligence, 27(8), 1226-1238.
    """
    
    def __init__(self, n_features_to_select=100, method='mrmr', 
                 discrete_target=True, random_state=None, verbose=1):
        """
        Initialize MRMR selector.
        
        Parameters:
        -----------
        n_features_to_select : int
            Number of features to select
        method : str
            'mrmr' or 'mifs' (Mutual Information Feature Selection)
        discrete_target : bool
            Whether target is discrete
        random_state : int or None
            Random seed
        verbose : int
            Verbosity level
        """
        self.n_features_to_select = n_features_to_select
        self.method = method.lower()
        self.discrete_target = discrete_target
        self.random_state = random_state
        self.verbose = verbose
        
        if self.method not in ['mrmr', 'mifs']:
            raise ValueError("method must be 'mrmr' or 'mifs'")
        
        self.selected_features = []
        self.feature_scores = {}
        self.mi_with_target = None
        self.mi_matrix = None
        
    def fit(self, X, y):
        """
        Select features using MRMR criterion.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
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
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Calculate MI with target
        if self.verbose >= 1:
            print("Calculating mutual information with target...")
        
        self.mi_with_target = mutual_info_classif(
            X, y,
            discrete_features=False,
            random_state=self.random_state,
            n_neighbors=3
        )
        
        # Precompute feature-feature MI matrix
        if self.verbose >= 1:
            print("Calculating feature-feature mutual information...")
        
        self.mi_matrix = np.zeros((n_features, n_features))
        
        # Compute upper triangle (symmetric matrix)
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
        self._greedy_forward_selection()
        
        return self
    
    def _greedy_forward_selection(self):
        """
        Perform greedy forward feature selection.
        """
        n_features = self.mi_with_target.shape[0]
        selected = []
        remaining = list(range(n_features))
        
        # Start with feature having max MI with target
        first_feature = np.argmax(self.mi_with_target)
        selected.append(first_feature)
        remaining.remove(first_feature)
        
        if self.verbose >= 1:
            print(f"\nStarting MRMR selection")
            print(f"Initial feature: {first_feature} (MI={self.mi_with_target[first_feature]:.4f})")
        
        # Progress bar
        pbar = tqdm(total=self.n_features_to_select - 1,
                   desc="MRMR selection",
                   disable=self.verbose < 1)
        
        while len(selected) < self.n_features_to_select and remaining:
            best_score = -np.inf
            best_feature = None
            
            for f in remaining:
                if self.method == 'mrmr':
                    score = self._mrmr_score(f, selected)
                elif self.method == 'mifs':
                    score = self._mifs_score(f, selected)
                
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
    
    def _mrmr_score(self, candidate, selected):
        """
        Calculate MRMR score.
        
        score = I(f; y) - 1/|S| Σ_{s in S} I(f; s)
        
        Parameters:
        -----------
        candidate : int
            Candidate feature index
        selected : list
            Selected feature indices
            
        Returns:
        --------
        score : float
            MRMR score
        """
        relevance = self.mi_with_target[candidate]
        
        if not selected:
            return relevance
        
        # Calculate average redundancy with selected features
        redundancy = 0
        for s in selected:
            redundancy += self.mi_matrix[candidate, s]
        
        redundancy /= len(selected)
        
        return relevance - redundancy
    
    def _mifs_score(self, candidate, selected):
        """
        Calculate MIFS score (with beta parameter).
        
        MIFS is similar to MRMR but includes a beta parameter to control
        the trade-off between relevance and redundancy.
        
        score = I(f; y) - β * Σ_{s in S} I(f; s)
        
        Parameters:
        -----------
        candidate : int
            Candidate feature index
        selected : list
            Selected feature indices
            
        Returns:
        --------
        score : float
            MIFS score
        """
        relevance = self.mi_with_target[candidate]
        
        if not selected:
            return relevance
        
        # Sum of redundancies (beta = 1/|S| for fair comparison)
        beta = 1.0 / len(selected)
        redundancy_sum = 0
        for s in selected:
            redundancy_sum += self.mi_matrix[candidate, s]
        
        return relevance - beta * redundancy_sum
    
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
        
        # Calculate importance scores
        importance_scores = np.zeros_like(self.mi_with_target)
        for idx, score in self.feature_scores.items():
            importance_scores[idx] = score
        
        return {
            'selected_features': self.selected_features.copy(),
            'mi_with_target': self.mi_with_target.copy(),
            'importance_scores': importance_scores,
            'feature_scores': self.feature_scores.copy(),
            'method': self.method
        }
    
    def calculate_redundancy(self):
        """
        Calculate redundancy among selected features.
        
        Returns:
        --------
        redundancy : float
            Average pairwise MI among selected features
        """
        if not self.selected_features:
            raise ValueError("No features selected")
        
        n_selected = len(self.selected_features)
        if n_selected < 2:
            return 0.0
        
        total_mi = 0
        count = 0
        
        for i in range(n_selected):
            for j in range(i + 1, n_selected):
                f1 = self.selected_features[i]
                f2 = self.selected_features[j]
                total_mi += self.mi_matrix[f1, f2]
                count += 1
        
        return total_mi / count if count > 0 else 0.0
    
    def calculate_relevance(self):
        """
        Calculate average relevance of selected features.
        
        Returns:
        --------
        relevance : float
            Average MI with target for selected features
        """
        if not self.selected_features:
            raise ValueError("No features selected")
        
        total_mi = 0
        for f in self.selected_features:
            total_mi += self.mi_with_target[f]
        
        return total_mi / len(self.selected_features)


# Convenience function
def select_features_mrmr(X, y, n_features=100, method='mrmr', return_scores=False):
    """
    Quick MRMR feature selection.
    
    Parameters:
    -----------
    X : array-like
        Input features
    y : array-like
        Target values
    n_features : int
        Number of features to select
    method : str
        'mrmr' or 'mifs'
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
    selector = MaxRelevanceMinRedundancy(
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