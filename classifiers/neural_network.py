"""
Support Vector Machine classifier optimized for microarray data.

Based on the paper's hyperparameter tuning:
- C parameter: 0.1-10
- Kernel: RBF
- Gamma: scale or auto
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SVMClassifier:
    """
    Support Vector Machine classifier for microarray data.
    
    Optimized with hyperparameters from the paper:
    - C: 0.1-10
    - Kernel: RBF
    - Gamma: scale or auto
    """
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3,
                 coef0=0.0, shrinking=True, probability=True,
                 tol=1e-3, cache_size=200, random_state=42,
                 verbose=0):
        """
        Initialize SVM classifier.
        
        Parameters (optimized based on paper):
        -----------
        C : float
            Regularization parameter (0.1-10)
        kernel : str
            Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
        gamma : str or float
            Kernel coefficient
        degree : int
            Degree for polynomial kernel
        coef0 : float
            Independent term in kernel
        shrinking : bool
            Use shrinking heuristic
        probability : bool
            Enable probability estimates
        tol : float
            Tolerance for stopping criterion
        cache_size : float
            Cache size in MB
        random_state : int
            Random seed
        verbose : int
            Verbosity level
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        
        # Performance metrics
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
    def fit(self, X, y, sample_weight=None):
        """
        Train SVM model.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
        sample_weight : array-like or None
            Sample weights
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Set random seed
        np.random.seed(self.random_state)
        
        # Scale features (important for SVM)
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and train model
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            shrinking=self.shrinking,
            probability=self.probability,
            tol=self.tol,
            cache_size=self.cache_size,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        
        # Calculate training metrics
        y_pred = self.predict(X)
        self.train_metrics = self._calculate_metrics(y, y_pred)
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : array-like
            Input features
            
        Returns:
        --------
        predictions : array
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like
            Input features
            
        Returns:
        --------
        probabilities : array
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if not self.probability:
            raise ValueError("Model was trained with probability=False")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict_decision_function(self, X):
        """
        Predict decision function values.
        
        Parameters:
        -----------
        X : array-like
            Input features
            
        Returns:
        --------
        decision_values : array
            Decision function values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X : array-like
            Test features
        y : array-like
            True labels
            
        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC if probability estimates are available
        if self.probability and len(np.unique(y)) == 2:
            try:
                y_proba = self.predict_proba(X)[:, 1]
                metrics['auc'] = roc_auc_score(y, y_proba)
            except:
                metrics['auc'] = 0.0
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
        
        # Support vectors information
        if hasattr(self.model, 'support_vectors_'):
            metrics['n_support_vectors'] = len(self.model.support_vectors_)
            metrics['support_vector_indices'] = self.model.support_.tolist()
        
        # Store test metrics
        self.test_metrics = metrics.copy()
        
        return metrics
    
    def cross_validate(self, X, y, n_splits=5, feature_selector=None,
                      return_fold_results=False):
        """
        Perform k-fold cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            Target labels
        n_splits : int
            Number of folds
        feature_selector : object
            Feature selection object
        return_fold_results : bool
            Whether to return detailed fold results
            
        Returns:
        --------
        cv_results : dict
            Cross-validation results
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=self.random_state)
        
        cv_results = {
            'fold_accuracies': [],
            'fold_precisions': [],
            'fold_recalls': [],
            'fold_f1s': [],
            'fold_aucs': [],
            'selected_features': [],
            'fold_models': [] if return_fold_results else None,
            'fold_scalers': [] if return_fold_results else None
        }
        
        fold = 1
        for train_idx, val_idx in skf.split(X, y):
            if self.verbose >= 1:
                print(f"\nFold {fold}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Apply feature selection if provided
            if feature_selector is not None:
                feature_selector.fit(X_train, y_train)
                X_train_selected = feature_selector.transform(X_train)
                X_val_selected = feature_selector.transform(X_val)
                cv_results['selected_features'].append(feature_selector.selected_features)
            else:
                X_train_selected = X_train
                X_val_selected = X_val
            
            # Create new scaler for this fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_val_scaled = scaler.transform(X_val_selected)
            
            # Create and train model
            model = SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
                shrinking=self.shrinking,
                probability=self.probability,
                tol=self.tol,
                cache_size=self.cache_size,
                random_state=self.random_state,
                verbose=self.verbose
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Store model and scaler if requested
            if return_fold_results:
                cv_results['fold_models'].append(model)
                cv_results['fold_scalers'].append(scaler)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val_scaled)
            fold_metrics = self._calculate_metrics(y_val, y_pred)
            
            # Add AUC if available
            if self.probability and len(np.unique(y_val)) == 2:
                try:
                    y_proba = model.predict_proba(X_val_scaled)[:, 1]
                    fold_metrics['auc'] = roc_auc_score(y_val, y_proba)
                    cv_results['fold_aucs'].append(fold_metrics['auc'])
                except:
                    fold_metrics['auc'] = 0.0
            
            cv_results['fold_accuracies'].append(fold_metrics['accuracy'])
            cv_results['fold_precisions'].append(fold_metrics['precision'])
            cv_results['fold_recalls'].append(fold_metrics['recall'])
            cv_results['fold_f1s'].append(fold_metrics['f1'])
            
            if self.verbose >= 1:
                print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
                print(f"  F1-score: {fold_metrics['f1']:.4f}")
            
            fold += 1
        
        # Calculate statistics
        cv_results['mean_accuracy'] = np.mean(cv_results['fold_accuracies'])
        cv_results['std_accuracy'] = np.std(cv_results['fold_accuracies'])
        cv_results['mean_f1'] = np.mean(cv_results['fold_f1s'])
        cv_results['std_f1'] = np.std(cv_results['fold_f1s'])
        
        if cv_results['fold_aucs']:
            cv_results['mean_auc'] = np.mean(cv_results['fold_aucs'])
            cv_results['std_auc'] = np.std(cv_results['fold_aucs'])
        
        if self.verbose >= 1:
            print(f"\nCross-validation results:")
            print(f"  Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
            print(f"  Mean F1-score: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
            if 'mean_auc' in cv_results:
                print(f"  Mean AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        
        return cv_results
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate performance metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
            
        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def tune_hyperparameters(self, X, y, param_grid=None, cv=5, scoring='accuracy'):
        """
        Tune hyperparameters using grid search.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
        param_grid : dict or None
            Parameter grid
        cv : int
            Cross-validation folds
        scoring : str
            Scoring metric
            
        Returns:
        --------
        best_params : dict
            Best parameters
        """
        if param_grid is None:
            # Default parameter grid based on paper
            param_grid = {
                'C': [0.1, 0.5, 1.0, 5.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'degree': [2, 3, 4]  # for polynomial kernel
            }
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create model for grid search
        svm = SVC(
            probability=self.probability,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            svm,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=self.verbose
        )
        
        grid_search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.C = grid_search.best_params_.get('C', self.C)
        self.kernel = grid_search.best_params_.get('kernel', self.kernel)
        self.gamma = grid_search.best_params_.get('gamma', self.gamma)
        self.degree = grid_search.best_params_.get('degree', self.degree)
        
        # Train final model with best parameters
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_, grid_search.best_score_
    
    def get_support_vectors(self):
        """
        Get support vectors.
        
        Returns:
        --------
        support_vectors : array or None
            Support vectors if available
        """
        if self.model is None or not hasattr(self.model, 'support_vectors_'):
            return None
        
        # Inverse transform to get original scale
        support_vectors = self.model.support_vectors_
        if hasattr(self.scaler, 'inverse_transform'):
            try:
                return self.scaler.inverse_transform(support_vectors)
            except:
                return support_vectors
        else:
            return support_vectors
    
    def get_dual_coefficients(self):
        """
        Get dual coefficients.
        
        Returns:
        --------
        dual_coefficients : array or None
            Dual coefficients if available
        """
        if self.model is None or not hasattr(self.model, 'dual_coef_'):
            return None
        
        return self.model.dual_coef_
    
    def save_model(self, filepath):
        """
        Save model to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save model
        """
        import joblib
        if self.model:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, filepath)
    
    def load_model(self, filepath):
        """
        Load model from file.
        
        Parameters:
        -----------
        filepath : str
            Path to model file
        """
        import joblib
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
    
    def __str__(self):
        """String representation."""
        return (f"SVMClassifier("
                f"C={self.C}, "
                f"kernel='{self.kernel}', "
                f"gamma={self.gamma})")