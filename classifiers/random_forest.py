"""
Random Forest classifier optimized for microarray data.

Based on the paper's hyperparameter tuning:
- Number of trees: 100-500
- Max depth: 10-30
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

class RandomForestClassifier:
    """
    Random Forest classifier for microarray data.
    
    Optimized with hyperparameters from the paper:
    - n_estimators: 100-500
    - max_depth: 10-30
    """
    
    def __init__(self, n_estimators=500, max_depth=20,
                 min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features='sqrt',
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, random_state=42,
                 n_jobs=-1, verbose=0):
        """
        Initialize Random Forest classifier.
        
        Parameters (optimized based on paper):
        -----------
        n_estimators : int
            Number of trees (100-500)
        max_depth : int
            Maximum tree depth (10-30)
        min_samples_split : int or float
            Minimum samples to split node
        min_samples_leaf : int or float
            Minimum samples in leaf node
        min_weight_fraction_leaf : float
            Minimum weighted fraction in leaf
        max_features : str, int or float
            Number of features to consider
        max_leaf_nodes : int or None
            Maximum leaf nodes
        min_impurity_decrease : float
            Minimum impurity decrease
        bootstrap : bool
            Use bootstrap samples
        oob_score : bool
            Use out-of-bag samples
        random_state : int
            Random seed
        n_jobs : int
            Number of parallel jobs
        verbose : int
            Verbosity level
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Initialize model
        self.model = None
        self.feature_importances_ = None
        
        # Performance metrics
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
    def fit(self, X, y, sample_weight=None):
        """
        Train Random Forest model.
        
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
        
        # Create and train model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        self.model.fit(X, y, sample_weight=sample_weight)
        
        # Get feature importance
        self.feature_importances_ = self.model.feature_importances_
        
        # Calculate training metrics
        y_pred = self.predict(X)
        self.train_metrics = self._calculate_metrics(y, y_pred)
        
        # OOB score if enabled
        if self.oob_score:
            self.train_metrics['oob_score'] = self.model.oob_score_
        
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
        
        return self.model.predict(X)
    
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
        
        return self.model.predict_proba(X)
    
    def predict_log_proba(self, X):
        """
        Predict class log-probabilities.
        
        Parameters:
        -----------
        X : array-like
            Input features
            
        Returns:
        --------
        log_probabilities : array
            Class log-probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.model.predict_log_proba(X)
    
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
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC for binary classification
        if len(np.unique(y)) == 2:
            try:
                metrics['auc'] = roc_auc_score(y, y_proba[:, 1])
            except:
                metrics['auc'] = 0.0
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
        
        # Feature importance
        if self.feature_importances_ is not None:
            metrics['feature_importance'] = self.feature_importances_.tolist()
        
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
            'fold_importances': []
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
            
            # Create and train model
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
            
            model.fit(X_train_selected, y_train)
            
            # Store model if requested
            if return_fold_results:
                cv_results['fold_models'].append(model)
            
            # Store feature importances
            cv_results['fold_importances'].append(model.feature_importances_)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val_selected)
            fold_metrics = self._calculate_metrics(y_val, y_pred)
            
            # Add AUC if available
            if len(np.unique(y_val)) == 2:
                try:
                    y_proba = model.predict_proba(X_val_selected)[:, 1]
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
        
        # Calculate average feature importance across folds
        if cv_results['fold_importances']:
            avg_importance = np.mean(cv_results['fold_importances'], axis=0)
            cv_results['mean_feature_importance'] = avg_importance.tolist()
        
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
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [10, 15, 20, 25, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        
        # Create model for grid search
        rf = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.n_estimators = grid_search.best_params_.get('n_estimators', self.n_estimators)
        self.max_depth = grid_search.best_params_.get('max_depth', self.max_depth)
        self.min_samples_split = grid_search.best_params_.get('min_samples_split', self.min_samples_split)
        self.min_samples_leaf = grid_search.best_params_.get('min_samples_leaf', self.min_samples_leaf)
        self.max_features = grid_search.best_params_.get('max_features', self.max_features)
        
        # Train final model with best parameters
        self.model = grid_search.best_estimator_
        self.feature_importances_ = self.model.feature_importances_
        
        return grid_search.best_params_, grid_search.best_score_
    
    def get_feature_importance(self):
        """
        Get feature importance.
        
        Returns:
        --------
        importance : array
            Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.model.feature_importances_
    
    def get_tree_depths(self):
        """
        Get depths of all trees.
        
        Returns:
        --------
        depths : list
            Tree depths
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return [estimator.tree_.max_depth for estimator in self.model.estimators_]
    
    def get_tree_leaf_counts(self):
        """
        Get leaf counts of all trees.
        
        Returns:
        --------
        leaf_counts : list
            Tree leaf counts
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return [estimator.tree_.n_leaves for estimator in self.model.estimators_]
    
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
            joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """
        Load model from file.
        
        Parameters:
        -----------
        filepath : str
            Path to model file
        """
        import joblib
        self.model = joblib.load(filepath)
        self.feature_importances_ = self.model.feature_importances_
    
    def __str__(self):
        """String representation."""
        return (f"RandomForestClassifier("
                f"n_estimators={self.n_estimators}, "
                f"max_depth={self.max_depth}, "
                f"max_features={self.max_features})")