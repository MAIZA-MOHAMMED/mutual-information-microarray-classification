"""
XGBoost classifier optimized for microarray data.

Based on the paper's hyperparameter tuning:
- Number of estimators: 100-300
- Max depth: 3-10
- Learning rate: 0.01-0.3
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

class XGBoostClassifier:
    """
    XGBoost classifier for microarray data.
    
    Optimized with hyperparameters from the paper:
    - n_estimators: 100-300
    - max_depth: 3-10  
    - learning_rate: 0.01-0.3
    """
    
    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0,
                 reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=0):
        """
        Initialize XGBoost classifier.
        
        Parameters (optimized based on paper):
        -----------
        n_estimators : int
            Number of trees (100-300)
        max_depth : int
            Maximum tree depth (3-10)
        learning_rate : float
            Learning rate (0.01-0.3)
        subsample : float
            Subsample ratio of training instances
        colsample_bytree : float
            Subsample ratio of columns
        reg_alpha : float
            L1 regularization
        reg_lambda : float
            L2 regularization
        random_state : int
            Random seed
        n_jobs : int
            Number of parallel jobs
        verbose : int
            Verbosity level
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Initialize model
        self.model = None
        self.feature_importances_ = None
        self.best_iteration = None
        
        # Performance metrics
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
    def fit(self, X, y, X_val=None, y_val=None, eval_metric='logloss',
            early_stopping_rounds=50, verbose_eval=None):
        """
        Train XGBoost model.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
        X_val, y_val : array-like or None
            Validation data
        eval_metric : str
            Evaluation metric
        early_stopping_rounds : int
            Early stopping rounds
        verbose_eval : int or None
            Verbosity for evaluation
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Set random seed
        np.random.seed(self.random_state)
        
        # Convert to DMatrix for efficiency
        dtrain = xgb.DMatrix(X, label=y)
        
        # Parameters
        params = {
            'objective': 'binary:logistic' if len(np.unique(y)) == 2 else 'multi:softprob',
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbosity': 0
        }
        
        if len(np.unique(y)) > 2:
            params['num_class'] = len(np.unique(y))
        
        # Training with validation if provided
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            
            evals = [(dtrain, 'train'), (dval, 'val')]
            evals_result = {}
            
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                evals_result=evals_result,
                verbose_eval=verbose_eval if verbose_eval is not None else self.verbose
            )
            
            self.best_iteration = self.model.best_iteration
            
            # Store training history
            self.train_metrics = {
                'loss': evals_result['train'][eval_metric][-1],
                'iterations': len(evals_result['train'][eval_metric])
            }
            
            self.val_metrics = {
                'val_loss': evals_result['val'][eval_metric][-1],
                'val_iterations': len(evals_result['val'][eval_metric])
            }
            
        else:
            # Train without validation
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                verbose_eval=verbose_eval if verbose_eval is not None else self.verbose
            )
        
        # Get feature importance
        self.feature_importances_ = self.model.get_score(importance_type='gain')
        
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
        
        dtest = xgb.DMatrix(X)
        pred_proba = self.model.predict(dtest)
        
        if pred_proba.ndim == 1 or pred_proba.shape[1] == 1:
            # Binary classification
            predictions = (pred_proba > 0.5).astype(int)
        else:
            # Multi-class classification
            predictions = np.argmax(pred_proba, axis=1)
        
        return predictions
    
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
        
        dtest = xgb.DMatrix(X)
        proba = self.model.predict(dtest)
        
        # For binary classification, ensure 2D output
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        
        return proba
    
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
        if self.feature_importances_:
            metrics['feature_importance'] = self.feature_importances_
        
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
            'fold_models': [] if return_fold_results else None
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
            
            # Train model
            self.fit(X_train_selected, y_train, X_val_selected, y_val,
                    verbose_eval=False)
            
            # Store model if requested
            if return_fold_results:
                cv_results['fold_models'].append(self.model.copy())
            
            # Evaluate on validation set
            fold_metrics = self.evaluate(X_val_selected, y_val)
            
            cv_results['fold_accuracies'].append(fold_metrics['accuracy'])
            cv_results['fold_precisions'].append(fold_metrics['precision'])
            cv_results['fold_recalls'].append(fold_metrics['recall'])
            cv_results['fold_f1s'].append(fold_metrics['f1'])
            
            if 'auc' in fold_metrics:
                cv_results['fold_aucs'].append(fold_metrics['auc'])
            
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
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'n_estimators': [100, 200, 300],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        
        # Create scikit-learn compatible XGBoost classifier
        if len(np.unique(y)) == 2:
            xgb_sklearn = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0
            )
        else:
            xgb_sklearn = xgb.XGBClassifier(
                objective='multi:softprob',
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0
            )
        
        # Perform grid search
        grid_search = GridSearchCV(
            xgb_sklearn,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.max_depth = grid_search.best_params_.get('max_depth', self.max_depth)
        self.learning_rate = grid_search.best_params_.get('learning_rate', self.learning_rate)
        self.n_estimators = grid_search.best_params_.get('n_estimators', self.n_estimators)
        self.subsample = grid_search.best_params_.get('subsample', self.subsample)
        self.colsample_bytree = grid_search.best_params_.get('colsample_bytree', self.colsample_bytree)
        
        # Train final model with best parameters
        self.fit(X, y)
        
        return grid_search.best_params_, grid_search.best_score_
    
    def get_feature_importance(self, importance_type='gain'):
        """
        Get feature importance.
        
        Parameters:
        -----------
        importance_type : str
            Type of importance: 'gain', 'weight', 'cover'
            
        Returns:
        --------
        importance : dict
            Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.model.get_score(importance_type=importance_type)
    
    def save_model(self, filepath):
        """
        Save model to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save model
        """
        if self.model:
            self.model.save_model(filepath)
    
    def load_model(self, filepath):
        """
        Load model from file.
        
        Parameters:
        -----------
        filepath : str
            Path to model file
        """
        self.model = xgb.Booster()
        self.model.load_model(filepath)
    
    def __str__(self):
        """String representation."""
        return (f"XGBoostClassifier("
                f"n_estimators={self.n_estimators}, "
                f"max_depth={self.max_depth}, "
                f"learning_rate={self.learning_rate})")