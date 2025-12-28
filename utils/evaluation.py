"""
Comprehensive evaluation utilities for microarray classification.
Includes metrics calculation, statistical tests, and result analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import scipy.stats as stats
import warnings
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive evaluator for microarray classification models.
    """
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize evaluator.
        
        Parameters:
        -----------
        n_splits : int
            Number of cross-validation folds
        random_state : int
            Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.results = {}
        self.best_models = {}
        
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, 
                       feature_selector=None, scoring: str = 'accuracy') -> Dict:
        """
        Perform cross-validation with optional feature selection.
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to evaluate
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        feature_selector : object, optional
            Feature selector with fit/transform methods
        scoring : str
            Scoring metric
            
        Returns:
        --------
        cv_results : Dict
            Cross-validation results
        """
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                            random_state=self.random_state)
        
        cv_results = {
            'fold_scores': [],
            'fold_predictions': [],
            'fold_true_labels': [],
            'fold_selected_features': [],
            'fold_training_time': [],
            'fold_inference_time': [],
            'models': []
        }
        
        import time
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            logger.info(f"Processing fold {fold}/{self.n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Feature selection if provided
            if feature_selector is not None:
                start_time = time.time()
                feature_selector.fit(X_train, y_train)
                X_train_selected = feature_selector.transform(X_train)
                X_val_selected = feature_selector.transform(X_val)
                fs_time = time.time() - start_time
                
                selected_features = feature_selector.selected_features
                cv_results['fold_selected_features'].append(selected_features)
            else:
                X_train_selected = X_train
                X_val_selected = X_val
                fs_time = 0
            
            # Train model
            start_time = time.time()
            model.fit(X_train_selected, y_train)
            training_time = time.time() - start_time
            
            # Predict
            start_time = time.time()
            y_pred = model.predict(X_val_selected)
            inference_time = time.time() - start_time
            
            # Calculate score
            if scoring == 'accuracy':
                score = accuracy_score(y_val, y_pred)
            elif scoring == 'f1':
                score = f1_score(y_val, y_pred, average='weighted')
            elif scoring == 'precision':
                score = precision_score(y_val, y_pred, average='weighted')
            elif scoring == 'recall':
                score = recall_score(y_val, y_pred, average='weighted')
            else:
                score = accuracy_score(y_val, y_pred)
            
            # Store results
            cv_results['fold_scores'].append(score)
            cv_results['fold_predictions'].append(y_pred)
            cv_results['fold_true_labels'].append(y_val)
            cv_results['fold_training_time'].append(training_time + fs_time)
            cv_results['fold_inference_time'].append(inference_time)
            cv_results['models'].append(model)
            
            logger.info(f"Fold {fold}: {scoring} = {score:.4f}, "
                       f"Training time = {training_time + fs_time:.2f}s")
        
        # Calculate summary statistics
        cv_results['mean_score'] = np.mean(cv_results['fold_scores'])
        cv_results['std_score'] = np.std(cv_results['fold_scores'])
        cv_results['mean_training_time'] = np.mean(cv_results['fold_training_time'])
        cv_results['mean_inference_time'] = np.mean(cv_results['fold_inference_time'])
        
        logger.info(f"Cross-validation completed: "
                   f"Mean {scoring} = {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        
        return cv_results
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                       X_train: Optional[np.ndarray] = None,
                       y_train: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive evaluation of a trained model.
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained model
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        X_train, y_train : np.ndarray, optional
            Training data for additional metrics
            
        Returns:
        --------
        metrics : Dict
            Comprehensive evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'kappa': cohen_kappa_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
        }
        
        # AUC if probabilities available
        if y_pred_proba is not None:
            n_classes = y_pred_proba.shape[1]
            if n_classes == 2:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                try:
                    metrics['auc_macro'] = roc_auc_score(y_test, y_pred_proba, 
                                                        multi_class='ovr', 
                                                        average='macro')
                except:
                    metrics['auc_macro'] = np.nan
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        if len(np.unique(y_test)) <= 10:  # Only for reasonable number of classes
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            metrics['per_class'] = report
        
        # Model complexity metrics
        if hasattr(model, 'n_features_in_'):
            metrics['n_features'] = model.n_features_in_
        
        if hasattr(model, 'coef_'):
            if isinstance(model.coef_, np.ndarray):
                metrics['n_nonzero_coef'] = np.sum(np.abs(model.coef_) > 1e-6)
        
        # Training metrics if training data provided
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
            metrics['train_f1'] = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
            
            # Calculate generalization gap
            metrics['generalization_gap'] = metrics['train_accuracy'] - metrics['accuracy']
        
        logger.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def compare_models(self, models_dict: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                      feature_selectors: Optional[Dict] = None) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.
        
        Parameters:
        -----------
        models_dict : Dict
            Dictionary of model names and instances
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        feature_selectors : Dict, optional
            Dictionary of feature selectors for each model
            
        Returns:
        --------
        comparison_df : pd.DataFrame
            Comparison results
        """
        results = []
        
        for model_name, model in models_dict.items():
            logger.info(f"Evaluating model: {model_name}")
            
            # Get feature selector for this model
            feature_selector = (feature_selectors[model_name] 
                              if feature_selectors and model_name in feature_selectors 
                              else None)
            
            # Cross-validate
            cv_results = self.cross_validate(
                model, X, y, 
                feature_selector=feature_selector,
                scoring='accuracy'
            )
            
            # Store results
            model_results = {
                'model': model_name,
                'mean_accuracy': cv_results['mean_score'],
                'std_accuracy': cv_results['std_score'],
                'mean_training_time': cv_results['mean_training_time'],
                'mean_inference_time': cv_results['mean_inference_time'],
                'n_features_selected': (
                    len(feature_selector.selected_features) 
                    if feature_selector else X.shape[1]
                )
            }
            
            results.append(model_results)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('mean_accuracy', ascending=False)
        
        # Statistical significance testing
        if len(results) > 1:
            self._add_statistical_significance(comparison_df, models_dict, X, y, feature_selectors)
        
        logger.info("Model comparison completed")
        print("\nModel Comparison Results:")
        print(comparison_df.to_string())
        
        return comparison_df
    
    def _add_statistical_significance(self, df: pd.DataFrame, models_dict: Dict,
                                     X: np.ndarray, y: np.ndarray,
                                     feature_selectors: Optional[Dict]) -> None:
        """Add statistical significance comparisons."""
        from scipy import stats
        
        # Get fold scores for each model
        fold_scores = {}
        for model_name in models_dict.keys():
            model = models_dict[model_name]
            feature_selector = (feature_selectors[model_name] 
                              if feature_selectors and model_name in feature_selectors 
                              else None)
            
            cv_results = self.cross_validate(
                model, X, y, 
                feature_selector=feature_selector,
                scoring='accuracy'
            )
            fold_scores[model_name] = cv_results['fold_scores']
        
        # Perform pairwise t-tests
        comparisons = []
        model_names = list(models_dict.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(
                    fold_scores[model1], 
                    fold_scores[model2]
                )
                
                # Determine significance
                significance = ''
                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'
                
                comparisons.append({
                    'comparison': f'{model1} vs {model2}',
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': significance
                })
        
        # Create comparison table
        comparison_table = pd.DataFrame(comparisons)
        self.significance_table = comparison_table
        
        logger.info("Statistical significance tests completed")
    
    def save_results(self, results: Dict, filename: str = None) -> None:
        """
        Save evaluation results to file.
        
        Parameters:
        -----------
        results : Dict
            Results to save
        filename : str, optional
            Output filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        import json
        import numpy as np
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                return super(NumpyEncoder, self).default(obj)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Results saved to {filename}")
    
    def load_results(self, filename: str) -> Dict:
        """
        Load evaluation results from file.
        
        Parameters:
        -----------
        filename : str
            Input filename
            
        Returns:
        --------
        results : Dict
            Loaded results
        """
        import json
        
        with open(filename, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from {filename}")
        return results


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                  y_pred_proba: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities
        
    Returns:
    --------
    metrics : Dict
        Comprehensive metrics dictionary
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Additional classification metrics
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Calculate derived metrics from confusion matrix
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
    
    # AUC if probabilities available
    if y_pred_proba is not None:
        try:
            n_classes = y_pred_proba.shape[1]
            if n_classes == 2:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['auc_macro'] = roc_auc_score(y_true, y_pred_proba, 
                                                    multi_class='ovr', 
                                                    average='macro')
                metrics['auc_weighted'] = roc_auc_score(y_true, y_pred_proba, 
                                                       multi_class='ovr', 
                                                       average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {str(e)}")
    
    # Per-class metrics for multi-class
    unique_classes = np.unique(y_true)
    if len(unique_classes) <= 10:  # Only for reasonable number of classes
        per_class_metrics = {}
        for cls in unique_classes:
            cls_idx = y_true == cls
            if cls_idx.any():
                per_class_metrics[f'class_{cls}_precision'] = precision_score(
                    y_true == cls, y_pred == cls, zero_division=0
                )
                per_class_metrics[f'class_{cls}_recall'] = recall_score(
                    y_true == cls, y_pred == cls, zero_division=0
                )
                per_class_metrics[f'class_{cls}_f1'] = f1_score(
                    y_true == cls, y_pred == cls, zero_division=0
                )
        metrics.update(per_class_metrics)
    
    return metrics


def perform_statistical_tests(results_dict: Dict, baseline_model: str = None) -> Dict:
    """
    Perform statistical significance tests between models.
    
    Parameters:
    -----------
    results_dict : Dict
        Dictionary with model names as keys and fold scores as values
    baseline_model : str, optional
        Name of baseline model for comparison
        
    Returns:
    --------
    stats_results : Dict
        Statistical test results
    """
    from scipy import stats
    
    stats_results = {}
    
    # Get all model names
    model_names = list(results_dict.keys())
    
    if baseline_model is None:
        baseline_model = model_names[0]
    
    # Perform tests
    for model in model_names:
        if model != baseline_model:
            # Check if we have fold scores
            if 'fold_scores' in results_dict[model] and 'fold_scores' in results_dict[baseline_model]:
                baseline_scores = results_dict[baseline_model]['fold_scores']
                model_scores = results_dict[model]['fold_scores']
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(baseline_scores, model_scores)
                
                # Wilcoxon signed-rank test (non-parametric)
                try:
                    w_stat, w_p_value = stats.wilcoxon(baseline_scores, model_scores)
                except:
                    w_stat, w_p_value = np.nan, np.nan
                
                stats_results[f'{baseline_model}_vs_{model}'] = {
                    't_test': {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant_0.05': p_value < 0.05,
                        'significant_0.01': p_value < 0.01,
                        'significant_0.001': p_value < 0.001
                    },
                    'wilcoxon': {
                        'statistic': w_stat,
                        'p_value': w_p_value,
                        'significant_0.05': w_p_value < 0.05 if not np.isnan(w_p_value) else False
                    }
                }
    
    return stats_results


def calculate_feature_importance(model, feature_names: List[str] = None,
                                method: str = 'default') -> pd.DataFrame:
    """
    Calculate and rank feature importance.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    feature_names : List[str], optional
        Feature names
    method : str
        Importance calculation method
        
    Returns:
    --------
    importance_df : pd.DataFrame
        Feature importance dataframe
    """
    importance_df = None
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
    elif hasattr(model, 'coef_'):
        # Linear models
        coef = model.coef_
        if len(coef.shape) == 1:
            importances = np.abs(coef)
        else:
            importances = np.mean(np.abs(coef), axis=0)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'coefficient': coef.flatten() if len(coef.shape) == 1 else coef.mean(axis=0)
        }).sort_values('importance', ascending=False)
    
    elif method == 'permutation':
        # Use permutation importance
        from sklearn.inspection import permutation_importance
        
        # Note: Need X and y for permutation importance
        logger.warning("Permutation importance requires X and y. Use permutation_importance directly.")
    
    if importance_df is not None:
        # Normalize importance to 0-100
        importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100)
        importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()
    
    return importance_df


def create_performance_report(metrics_dict: Dict, model_name: str = None) -> str:
    """
    Create a formatted performance report.
    
    Parameters:
    -----------
    metrics_dict : Dict
        Metrics dictionary
    model_name : str, optional
        Model name for report
        
    Returns:
    --------
    report : str
        Formatted report
    """
    report = []
    
    if model_name:
        report.append(f"Model: {model_name}")
        report.append("=" * 50)
    
    # Basic metrics
    report.append("\nBasic Metrics:")
    report.append("-" * 30)
    for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']:
        if metric in metrics_dict:
            report.append(f"{metric:20s}: {metrics_dict[metric]:.4f}")
    
    # Additional metrics
    report.append("\nAdditional Metrics:")
    report.append("-" * 30)
    for metric in ['mcc', 'kappa', 'balanced_accuracy', 'auc', 'auc_macro']:
        if metric in metrics_dict:
            report.append(f"{metric:20s}: {metrics_dict[metric]:.4f}")
    
    # Confusion matrix if available
    if 'confusion_matrix' in metrics_dict:
        report.append("\nConfusion Matrix:")
        report.append("-" * 30)
        cm = metrics_dict['confusion_matrix']
        if isinstance(cm, np.ndarray):
            cm_str = "\n".join(["  ".join(map(str, row)) for row in cm])
            report.append(cm_str)
    
    # Per-class metrics
    if 'per_class' in metrics_dict:
        report.append("\nPer-class Metrics:")
        report.append("-" * 30)
        per_class = metrics_dict['per_class']
        for cls, cls_metrics in per_class.items():
            if cls not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(cls_metrics, dict):
                report.append(f"\nClass {cls}:")
                for metric_name, value in cls_metrics.items():
                    if metric_name != 'support':
                        report.append(f"  {metric_name}: {value:.4f}")
    
    return "\n".join(report)


# Example usage function
def example_usage():
    """Example usage of evaluation utilities."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=200,
        n_features=1000,
        n_informative=50,
        n_redundant=100,
        random_state=42
    )
    
    # Create models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Initialize evaluator
    evaluator = ModelEvaluator(n_splits=5, random_state=42)
    
    # Compare models
    comparison_df = evaluator.compare_models(models, X, y)
    
    # Evaluate a single model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    # Create test set
    X_test, y_test = make_classification(
        n_samples=50,
        n_features=1000,
        n_informative=50,
        n_redundant=100,
        random_state=43
    )
    
    metrics = evaluator.evaluate_model(rf_model, X_test, y_test, X, y)
    
    # Create report
    report = create_performance_report(metrics, "Random Forest")
    print(report)
    
    return comparison_df, metrics


if __name__ == "__main__":
    # Run example
    comparison_df, metrics = example_usage()