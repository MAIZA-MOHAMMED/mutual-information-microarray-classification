"""
Mutual Information Feature Selector for High-Dimensional Microarray Data
Conceptual implementation based on the paper methodology
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

class MutualInformationSelector:
    """
    Feature selector using mutual information for microarray data
    """
    
    def __init__(self, n_features_to_select=100):
        """
        Initialize the feature selector
        
        Parameters:
        n_features_to_select (int): Number of top features to select
        """
        self.n_features_to_select = n_features_to_select
        self.selected_features = None
        self.feature_scores = None
        self.feature_ranking = None
    
    def fit(self, X, y):
        """
        Compute mutual information between each feature and target
        
        Parameters:
        X (np.ndarray): Microarray data matrix [n_samples, n_features]
        y (np.ndarray): Class labels
        
        Returns:
        self: Fitted selector instance
        """
        # Normalize data (mutual information is scale-invariant, but good practice)
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        
        # Compute mutual information scores
        self.feature_scores = mutual_info_classif(
            X_normalized, 
            y, 
            discrete_features=False,  # Microarray data is typically continuous
            random_state=42,
            n_neighbors=3  # Parameter for MI estimation
        )
        
        # Create feature ranking (highest score first)
        self.feature_ranking = np.argsort(self.feature_scores)[::-1]
        
        # Select top features
        self.selected_features = self.feature_ranking[:self.n_features_to_select]
        
        return self
    
    def transform(self, X):
        """
        Transform data using selected features
        
        Parameters:
        X (np.ndarray): Original data matrix
        
        Returns:
        np.ndarray: Data matrix with only selected features
        """
        if self.selected_features is None:
            raise ValueError("Selector must be fitted before transformation.")
        
        return X[:, self.selected_features]
    
    def fit_transform(self, X, y):
        """
        Fit and transform data in one step
        
        Parameters:
        X (np.ndarray): Original data matrix
        y (np.ndarray): Class labels
        
        Returns:
        np.ndarray: Transformed data matrix
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
        dict: Feature indices and their MI scores
        """
        return {
            'feature_indices': np.arange(len(self.feature_scores)),
            'mi_scores': self.feature_scores,
            'ranking': self.feature_ranking
        }

# Example comparison with other methods
def compare_feature_selection_methods(X, y, n_features=100):
    """
    Compare Mutual Information with other feature selection methods
    
    Parameters:
    X (np.ndarray): Data matrix
    y (np.ndarray): Labels
    n_features (int): Number of features to select
    
    Returns:
    dict: Results from different methods
    """
    from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    results = {}
    
    # 1. Mutual Information
    mi_selector = MutualInformationSelector(n_features_to_select=n_features)
    X_mi = mi_selector.fit_transform(X, y)
    results['Mutual_Information'] = {
        'selected_features': mi_selector.selected_features,
        'scores': mi_selector.feature_scores
    }
    
    # 2. ANOVA F-value
    fvalue_selector = SelectKBest(f_classif, k=n_features)
    X_f = fvalue_selector.fit_transform(X, y)
    results['ANOVA_F'] = {
        'selected_features': fvalue_selector.get_support(indices=True),
        'scores': fvalue_selector.scores_
    }
    
    # 3. Random Forest Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    rf_selected = np.argsort(rf_importance)[-n_features:]
    results['Random_Forest'] = {
        'selected_features': rf_selected,
        'scores': rf_importance
    }
    
    return results


# Example usage
if __name__ == "__main__":
    # Simulate microarray-like data
    n_samples = 150
    n_features = 10000  # High-dimensional like microarray
    n_informative = 50  # Only 50 truly informative features
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Create informative features with non-linear relationships
    for i in range(n_informative):
        if i % 2 == 0:
            # Linear relationship for some features
            X[:, i] = X[:, i] * 2 + np.random.randn(n_samples) * 0.5
        else:
            # Non-linear relationship for others
            X[:, i] = np.sin(X[:, i]) + np.random.randn(n_samples) * 0.5
    
    # Create labels based on informative features
    y = (np.sum(X[:, :n_informative], axis=1) > 0).astype(int)
    
    # Apply mutual information feature selection
    selector = MutualInformationSelector(n_features_to_select=100)
    X_selected = selector.fit_transform(X, y)
    
    print(f"Original data shape: {X.shape}")
    print(f"Selected data shape: {X_selected.shape}")
    print(f"Top 10 selected features: {selector.selected_features[:10]}")
    
    # Compare methods
    comparison = compare_feature_selection_methods(X, y, n_features=100)
    
    # Calculate overlap between methods
    mi_features = set(comparison['Mutual_Information']['selected_features'])
    anova_features = set(comparison['ANOVA_F']['selected_features'])
    overlap = len(mi_features.intersection(anova_features))
    
    print(f"\nOverlap between MI and ANOVA: {overlap} features")
    print(f"Jaccard similarity: {overlap/100:.2%}")
