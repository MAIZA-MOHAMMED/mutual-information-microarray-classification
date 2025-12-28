"""
Mutual Information Maximization (MIM) feature selection.

MIM selects features with highest individual mutual information with the target.
This is the simplest mutual information-based feature selection method.
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm
from .utils import estimate_mutual_information

class MutualInformationMaximization:
    """
    Mutual Information Maximization feature selector.
    
    Selects features based on their individual mutual information with the target.
    This method doesn't consider interactions between features.
    
    References:
    -----------
    Battiti, R. (1994). Using mutual information for selecting features 
    in supervised neural net learning. IEEE Transactions on neural networks, 5(4), 537-550.
    """
    
    def __init__(self, n_features_to_select=100, discrete_target=False, random_state=None):
        """
        Initialize MIM selector.
        
        Parameters:
        -----------
        n_features_to_select : int
            Number of features to select
        discrete_target : bool
            Whether the target variable is discrete
        random_state : int or None
            Random seed for reproducibility
        """
        self.n_features_to_select = n_features_to_select
        self.discrete_target = discrete_target
        self.random_state = random_state
        self.selected_features = None
        self.feature_scores = None
        self.feature_ranking = None
        
    def fit(self, X, y):
        """
        Compute mutual information scores and select features.
        
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
        
        # Estimate mutual information for each feature
        if self.discrete_target:
            # For discrete targets, use scikit-learn's implementation
            self.feature_scores = mutual_info_classif(
                X, y, 
                discrete_features=False,
                random_state=self.random_state,
                n_neighbors=3
            )
        else:
            # For continuous targets, use our implementation
            self.feature_scores = np.zeros(n_features)
            for i in tqdm(range(n_features), desc="Calculating MI scores"):
                self.feature_scores[i] = estimate_mutual_information(
                    X[:, i], y, 
                    discrete_x=False, 
                    discrete_y=False
                )
        
        # Rank features by MI score (descending)
        self.feature_ranking = np.argsort(self.feature_scores)[::-1]
        
        # Select top features
        self.selected_features = self.feature_ranking[:self.n_features_to_select]
        
        # Store scores for selected features
        self.selected_scores = self.feature_scores[self.selected_features]
        
        return self
    
    def transform(self, X):
        """
        Reduce X to selected features.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X_reduced : array-like, shape (n_samples, n_selected_features)
            Data with selected features
        """
        if self.selected_features is None:
            raise ValueError("Must call fit before transform")
        
        return X[:, self.selected_features]
    
    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        X_reduced : array-like, shape (n_samples, n_selected_features)
            Transformed data
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self, indices=True):
        """
        Get a mask or indices of selected features.
        
        Parameters:
        -----------
        indices : bool
            If True, return indices. If False, return boolean mask.
            
        Returns:
        --------
        support : array
            Indices or mask of selected features
        """
        if self.selected_features is None:
            raise ValueError("Must call fit first")
        
        if indices:
            return self.selected_features.copy()
        else:
            mask = np.zeros(len(self.feature_scores), dtype=bool)
            mask[self.selected_features] = True
            return mask
    
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
        --------
        importance : dict
            Dictionary with feature indices and their MI scores
        """
        if self.feature_scores is None:
            raise ValueError("Must call fit first")
        
        return {
            'feature_indices': np.arange(len(self.feature_scores)),
            'mi_scores': self.feature_scores.copy(),
            'ranking': self.feature_ranking.copy()
        }
    
    def score_features(self, X, y):
        """
        Score features without selecting them.
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            Target values
            
        Returns:
        --------
        scores : array
            MI scores for each feature
        """
        return mutual_info_classif(X, y, discrete_features=False)
    
    def get_selected_feature_names(self, feature_names=None):
        """
        Get names of selected features.
        
        Parameters:
        -----------
        feature_names : list or None
            List of feature names. If None, returns indices.
            
        Returns:
        --------
        selected_names : list
            Names of selected features
        """
        if feature_names is None:
            return self.selected_features.tolist()
        
        if len(feature_names) != len(self.feature_scores):
            raise ValueError(f"feature_names length ({len(feature_names)}) "
                           f"must match number of features ({len(self.feature_scores)})")
        
        return [feature_names[i] for i in self.selected_features]
    
    def __str__(self):
        """String representation."""
        if self.selected_features is None:
            return "MutualInformationMaximization (not fitted)"
        
        return (f"MutualInformationMaximization: "
                f"selected {len(self.selected_features)} features")
    
    def __repr__(self):
        """Detailed representation."""
        return (f"MutualInformationMaximization("
                f"n_features_to_select={self.n_features_to_select}, "
                f"discrete_target={self.discrete_target})")


# Convenience function for quick MIM selection
def select_features_mim(X, y, n_features=100, return_scores=False):
    """
    Quick MIM feature selection.
    
    Parameters:
    -----------
    X : array-like
        Input features
    y : array-like
        Target values
    n_features : int
        Number of features to select
    return_scores : bool
        Whether to return MI scores
        
    Returns:
    --------
    X_selected : array
        Selected features
    selected_indices : array
        Indices of selected features
    scores : array, optional
        MI scores if return_scores=True
    """
    selector = MutualInformationMaximization(n_features_to_select=n_features)
    X_selected = selector.fit_transform(X, y)
    
    if return_scores:
        return X_selected, selector.selected_features, selector.feature_scores
    else:
        return X_selected, selector.selected_features