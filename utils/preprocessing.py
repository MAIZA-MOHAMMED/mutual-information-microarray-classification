"""
Data preprocessing utilities for microarray data analysis.
Includes loading, cleaning, normalization, and feature engineering functions.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MicroarrayPreprocessor:
    """
    Comprehensive preprocessor for microarray data following the paper methodology.
    """
    
    def __init__(self, 
                 normalize_method: str = 'standard',
                 handle_missing: str = 'mean',
                 remove_constant: bool = True,
                 variance_threshold: float = 0.01,
                 random_state: int = 42):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        normalize_method : str
            Normalization method: 'standard', 'minmax', 'robust', or None
        handle_missing : str
            Missing value handling: 'mean', 'median', 'knn', or 'drop'
        remove_constant : bool
            Whether to remove constant features
        variance_threshold : float
            Remove features with variance below this threshold
        random_state : int
            Random seed for reproducibility
        """
        self.normalize_method = normalize_method
        self.handle_missing = handle_missing
        self.remove_constant = remove_constant
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        
        # Initialize transformers
        self.scaler = None
        self.imputer = None
        self.variance_filter = None
        self.label_encoder = None
        self.feature_names = None
        self.constant_features_mask = None
        
    def load_microarray_data(self, filepath: str, target_column: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load microarray data from various formats.
        
        Parameters:
        -----------
        filepath : str
            Path to data file (CSV, TSV, or TXT)
        target_column : str, optional
            Name of target column. If None, assumes last column is target.
            
        Returns:
        --------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target labels (n_samples,)
        """
        logger.info(f"Loading data from {filepath}")
        
        try:
            # Detect file format and load
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.tsv'):
                df = pd.read_csv(filepath, sep='\t')
            elif filepath.endswith('.txt'):
                df = pd.read_csv(filepath, sep='\s+')
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            logger.info(f"Loaded data shape: {df.shape}")
            
            # Identify target column
            if target_column is None:
                # Assume last column is target
                target_column = df.columns[-1]
                logger.info(f"Assuming target column: {target_column}")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Separate features and target
            X = df.drop(columns=[target_column]).values.astype(np.float32)
            y = df[target_column].values
            
            # Store feature names
            self.feature_names = df.drop(columns=[target_column]).columns.tolist()
            
            logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            logger.info(f"Feature names stored: {len(self.feature_names)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                   fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply complete preprocessing pipeline.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray, optional
            Target labels
        fit : bool
            Whether to fit transformers or just transform
            
        Returns:
        --------
        X_processed : np.ndarray
            Processed feature matrix
        y_processed : np.ndarray or None
            Processed target labels (if provided)
        """
        X_original_shape = X.shape
        logger.info(f"Starting preprocessing. Original shape: {X_original_shape}")
        
        # Step 1: Handle missing values
        X = self._handle_missing_values(X, fit)
        
        # Step 2: Remove constant features
        if self.remove_constant:
            X = self._remove_constant_features(X, fit)
        
        # Step 3: Apply variance threshold
        if self.variance_threshold > 0:
            X = self._apply_variance_threshold(X, fit)
        
        # Step 4: Normalize features
        X = self._normalize_features(X, fit)
        
        # Step 5: Encode labels if y is provided
        y_processed = None
        if y is not None:
            y_processed = self._encode_labels(y, fit)
        
        logger.info(f"Preprocessing complete. Final shape: {X.shape}")
        logger.info(f"Reduced from {X_original_shape[1]} to {X.shape[1]} features "
                   f"({(X_original_shape[1] - X.shape[1]) / X_original_shape[1] * 100:.1f}% reduction)")
        
        return X, y_processed
    
    def _handle_missing_values(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """Handle missing values in microarray data."""
        missing_count = np.isnan(X).sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values ({missing_count/X.size*100:.2f}%)")
            
            if self.handle_missing == 'drop':
                # Drop samples with missing values
                mask = ~np.isnan(X).any(axis=1)
                X = X[mask]
                logger.info(f"Dropped {len(mask) - mask.sum()} samples with missing values")
                
            elif self.handle_missing in ['mean', 'median']:
                # Impute missing values
                if fit or self.imputer is None:
                    strategy = 'mean' if self.handle_missing == 'mean' else 'median'
                    self.imputer = SimpleImputer(strategy=strategy)
                    self.imputer.fit(X)
                X = self.imputer.transform(X)
                logger.info(f"Imputed missing values using {self.handle_missing}")
                
            elif self.handle_missing == 'knn':
                # Use KNN imputation
                if fit or self.imputer is None:
                    self.imputer = KNNImputer(n_neighbors=5)
                    self.imputer.fit(X)
                X = self.imputer.transform(X)
                logger.info("Imputed missing values using KNN")
        else:
            logger.info("No missing values found")
            
        return X
    
    def _remove_constant_features(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """Remove constant or near-constant features."""
        variances = np.var(X, axis=0)
        constant_mask = variances <= 1e-10
        
        if fit:
            self.constant_features_mask = constant_mask
            self.feature_names = [name for name, is_const in zip(self.feature_names, constant_mask) 
                                if not is_const]
        
        if constant_mask.any():
            X = X[:, ~constant_mask]
            logger.info(f"Removed {constant_mask.sum()} constant features")
            
        return X
    
    def _apply_variance_threshold(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """Apply variance threshold filtering."""
        if fit or self.variance_filter is None:
            self.variance_filter = VarianceThreshold(threshold=self.variance_threshold)
            self.variance_filter.fit(X)
            
        X = self.variance_filter.transform(X)
        
        if fit:
            # Update feature names
            if self.feature_names:
                support_mask = self.variance_filter.get_support()
                self.feature_names = [name for name, keep in zip(self.feature_names, support_mask) 
                                    if keep]
        
        logger.info(f"Applied variance threshold ({self.variance_threshold})")
        return X
    
    def _normalize_features(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """Normalize features using specified method."""
        if self.normalize_method is None:
            return X
            
        if fit or self.scaler is None:
            if self.normalize_method == 'standard':
                self.scaler = StandardScaler()
            elif self.normalize_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.normalize_method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.normalize_method}")
            
            self.scaler.fit(X)
            
        X = self.scaler.transform(X)
        logger.info(f"Applied {self.normalize_method} normalization")
        
        return X
    
    def _encode_labels(self, y: np.ndarray, fit: bool) -> np.ndarray:
        """Encode categorical labels to numerical values."""
        if fit or self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            
        y_encoded = self.label_encoder.transform(y)
        
        # Log class distribution
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        class_info = ", ".join([f"Class {cls}: {count}" for cls, count in zip(unique_classes, class_counts)])
        logger.info(f"Encoded labels. Classes: {class_info}")
        
        return y_encoded
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        if self.feature_names is None:
            raise ValueError("Feature names not available. Load data first.")
        return self.feature_names
    
    def get_preprocessing_summary(self) -> Dict:
        """Get summary of preprocessing steps applied."""
        summary = {
            'normalization_method': self.normalize_method,
            'missing_value_handling': self.handle_missing,
            'constant_features_removed': self.remove_constant,
            'variance_threshold': self.variance_threshold,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'imputer_type': type(self.imputer).__name__ if self.imputer else None
        }
        return summary


def create_train_test_split(X: np.ndarray, y: np.ndarray, 
                           test_size: float = 0.2, 
                           stratify: bool = True,
                           random_state: int = 42) -> Tuple:
    """
    Create train-test split for microarray data.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    test_size : float
        Proportion of test samples
    stratify : bool
        Whether to preserve class distribution
    random_state : int
        Random seed
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : Tuple
        Split data
    """
    from sklearn.model_selection import train_test_split
    
    if stratify and len(np.unique(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y,
            random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
    
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    logger.info(f"Train class distribution: {np.unique(y_train, return_counts=True)}")
    logger.info(f"Test class distribution: {np.unique(y_test, return_counts=True)}")
    
    return X_train, X_test, y_train, y_test


def create_cross_validation_splits(X: np.ndarray, y: np.ndarray, 
                                  n_splits: int = 5,
                                  shuffle: bool = True,
                                  random_state: int = 42):
    """
    Create cross-validation splits for microarray data.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    n_splits : int
        Number of folds
    shuffle : bool
        Whether to shuffle data
    random_state : int
        Random seed
        
    Returns:
    --------
    folds : generator
        Generator yielding (train_idx, test_idx) for each fold
    """
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return skf.split(X, y)


def balance_dataset(X: np.ndarray, y: np.ndarray, 
                    method: str = 'oversample',
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance dataset to handle class imbalance.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    method : str
        Balancing method: 'oversample', 'undersample', or 'smote'
    random_state : int
        Random seed
        
    Returns:
    --------
    X_balanced, y_balanced : Tuple
        Balanced dataset
    """
    from collections import Counter
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    
    class_counts = Counter(y)
    logger.info(f"Original class distribution: {class_counts}")
    
    if method == 'oversample':
        sampler = RandomOverSampler(random_state=random_state)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'smote':
        sampler = SMOTE(random_state=random_state)
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    X_balanced, y_balanced = sampler.fit_resample(X, y)
    
    balanced_counts = Counter(y_balanced)
    logger.info(f"Balanced class distribution: {balanced_counts}")
    
    return X_balanced, y_balanced


def extract_top_features(X: np.ndarray, y: np.ndarray, 
                        feature_names: List[str], 
                        n_features: int = 100,
                        method: str = 'f_classif') -> Tuple[np.ndarray, List[str]]:
    """
    Extract top features using statistical tests.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    feature_names : List[str]
        List of feature names
    n_features : int
        Number of top features to select
    method : str
        Feature selection method: 'f_classif', 'mutual_info_classif', 'chi2'
        
    Returns:
    --------
    X_selected : np.ndarray
        Selected features
    selected_names : List[str]
        Names of selected features
    """
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
    
    if method == 'f_classif':
        selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
    elif method == 'mutual_info_classif':
        selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
    elif method == 'chi2':
        selector = SelectKBest(chi2, k=min(n_features, X.shape[1]))
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_names = [feature_names[i] for i in selected_indices]
    
    # Get feature scores
    scores = selector.scores_[selected_indices]
    
    logger.info(f"Selected top {len(selected_names)} features using {method}")
    logger.info(f"Top 10 features: {selected_names[:10]}")
    
    return X_selected, selected_names, scores


def create_data_summary(X: np.ndarray, y: np.ndarray, 
                       feature_names: Optional[List[str]] = None) -> Dict:
    """
    Create comprehensive summary of microarray dataset.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    feature_names : List[str], optional
        Feature names
        
    Returns:
    --------
    summary : Dict
        Dataset summary
    """
    summary = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'missing_values': np.isnan(X).sum(),
        'feature_stats': {
            'mean': np.mean(X, axis=0).tolist(),
            'std': np.std(X, axis=0).tolist(),
            'min': np.min(X, axis=0).tolist(),
            'max': np.max(X, axis=0).tolist()
        },
        'sparsity': (X == 0).sum() / X.size * 100 if X.size > 0 else 0
    }
    
    if feature_names:
        summary['feature_names'] = feature_names[:10]  # First 10 for brevity
        
    return summary


# Example usage function
def example_usage():
    """Example usage of preprocessing utilities."""
    # Create synthetic microarray-like data
    np.random.seed(42)
    n_samples = 100
    n_features = 1000
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Add some missing values
    mask = np.random.rand(*X.shape) < 0.01
    X[mask] = np.nan
    
    # Add some constant features
    X[:, 0] = 1.0  # Constant feature
    
    feature_names = [f'Gene_{i}' for i in range(n_features)]
    
    # Initialize preprocessor
    preprocessor = MicroarrayPreprocessor(
        normalize_method='standard',
        handle_missing='mean',
        remove_constant=True,
        variance_threshold=0.01
    )
    
    # Preprocess data
    X_processed, y_processed = preprocessor.preprocess(X, y, fit=True)
    
    # Get summary
    summary = preprocessor.get_preprocessing_summary()
    print("Preprocessing Summary:", summary)
    
    # Create train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(
        X_processed, y_processed, test_size=0.2
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Run example
    X_train, X_test, y_train, y_test = example_usage()