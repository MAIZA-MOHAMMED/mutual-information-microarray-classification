"""
Visualization utilities for microarray data analysis.
Includes plotting functions for data exploration, results visualization, and model interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import logging
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MicroarrayVisualizer:
    """
    Comprehensive visualizer for microarray data analysis.
    """
    
    def __init__(self, figsize: Tuple = (10, 8), dpi: int = 100):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        figsize : Tuple
            Default figure size
        dpi : int
            Figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", 10)
        
    def plot_dataset_summary(self, dataset_info: Dict, save_path: str = None) -> plt.Figure:
        """
        Plot dataset summary information.
        
        Parameters:
        -----------
        dataset_info : Dict
            Dataset information dictionary
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Generated figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Microarray Dataset Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Sample and feature counts
        datasets = list(dataset_info.keys())
        n_samples = [info.get('n_samples', 0) for info in dataset_info.values()]
        n_features = [info.get('n_features', 0) for info in dataset_info.values()]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, n_samples, width, label='Samples', alpha=0.7)
        axes[0, 0].bar(x + width/2, n_features, width, label='Features', alpha=0.7)
        axes[0, 0].set_xlabel('Dataset')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Samples vs Features')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(datasets, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Dimensionality ratio (p/n)
        pn_ratios = []
        for info in dataset_info.values():
            n_samples = info.get('n_samples', 1)
            n_features = info.get('n_features', 1)
            pn_ratios.append(n_features / n_samples)
        
        axes[0, 1].bar(datasets, pn_ratios, color='coral', alpha=0.7)
        axes[0, 1].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='p/n = 100')
        axes[0, 1].set_xlabel('Dataset')
        axes[0, 1].set_ylabel('Features/Samples Ratio')
        axes[0, 1].set_title('Dimensionality Challenge')
        axes[0, 1].set_xticklabels(datasets, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Class distribution
        class_imbalance = []
        for info in dataset_info.values():
            if 'class_distribution' in info:
                counts = list(info['class_distribution'].values())
                if counts:
                    imbalance = min(counts) / max(counts)
                    class_imbalance.append(imbalance)
            else:
                class_imbalance.append(1.0)
        
        axes[0, 2].bar(datasets, class_imbalance, color='lightgreen', alpha=0.7)
        axes[0, 2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Balanced=0.5')
        axes[0, 2].set_xlabel('Dataset')
        axes[0, 2].set_ylabel('Class Balance Ratio')
        axes[0, 2].set_title('Class Imbalance Analysis')
        axes[0, 2].set_xticklabels(datasets, rotation=45, ha='right')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Missing values
        missing_percent = []
        for info in dataset_info.values():
            n_samples = info.get('n_samples', 1)
            n_features = info.get('n_features', 1)
            missing = info.get('missing_values', 0)
            missing_percent.append(missing / (n_samples * n_features) * 100)
        
        axes[1, 0].bar(datasets, missing_percent, color='skyblue', alpha=0.7)
        axes[1, 0].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
        axes[1, 0].set_xlabel('Dataset')
        axes[1, 0].set_ylabel('Missing Values (%)')
        axes[1, 0].set_title('Missing Data Analysis')
        axes[1, 0].set_xticklabels(datasets, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Sparsity
        sparsity = [info.get('sparsity', 0) for info in dataset_info.values()]
        axes[1, 1].bar(datasets, sparsity, color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Dataset')
        axes[1, 1].set_ylabel('Sparsity (%)')
        axes[1, 1].set_title('Data Sparsity')
        axes[1, 1].set_xticklabels(datasets, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Complexity heatmap
        complexity_data = []
        for info in dataset_info.values():
            complexity_data.append([
                info.get('n_samples', 0),
                info.get('n_features', 0),
                info.get('n_features', 1) / max(info.get('n_samples', 1), 1),
                info.get('missing_values', 0) / max(info.get('n_samples', 1) * info.get('n_features', 1), 1) * 100,
                info.get('sparsity', 0)
            ])
        
        complexity_df = pd.DataFrame(complexity_data, index=datasets,
                                   columns=['Samples', 'Features', 'p/n Ratio', 'Missing%', 'Sparsity%'])
        complexity_normalized = (complexity_df - complexity_df.min()) / (complexity_df.max() - complexity_df.min())
        
        sns.heatmap(complexity_normalized.T, annot=True, fmt='.2f', cmap='YlOrRd',
                   cbar_kws={'label': 'Normalized Value'}, ax=axes[1, 2])
        axes[1, 2].set_title('Dataset Complexity Heatmap')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_feature_distribution(self, X: np.ndarray, y: np.ndarray = None,
                                 feature_names: List[str] = None,
                                 n_features: int = 10,
                                 save_path: str = None) -> plt.Figure:
        """
        Plot feature distribution analysis.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray, optional
            Target labels
        feature_names : List[str], optional
            Feature names
        n_features : int
            Number of features to plot
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Generated figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Feature Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall distribution
        axes[0, 0].hist(X.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Feature Values')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Overall Feature Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Distribution by class (if y provided)
        if y is not None and len(np.unique(y)) == 2:
            for class_label in np.unique(y):
                class_data = X[y == class_label].flatten()
                axes[0, 1].hist(class_data, bins=50, alpha=0.5, 
                               label=f'Class {class_label}', density=True)
            axes[0, 1].set_xlabel('Feature Values')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Distribution by Class')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            # Plot variance distribution instead
            variances = np.var(X, axis=0)
            axes[0, 1].hist(variances, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_xlabel('Feature Variance')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Feature Variance Distribution')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Top n_features with highest variance
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-n_features:]
        
        if feature_names is not None:
            top_names = [feature_names[i] for i in top_indices]
        else:
            top_names = [f'Feature {i}' for i in top_indices]
        
        top_variances = variances[top_indices]
        
        axes[0, 2].barh(range(n_features), top_variances, alpha=0.7, color='orange')
        axes[0, 2].set_yticks(range(n_features))
        axes[0, 2].set_yticklabels(top_names)
        axes[0, 2].set_xlabel('Variance')
        axes[0, 2].set_title(f'Top {n_features} Most Variable Features')
        axes[0, 2].grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Correlation matrix (subset)
        np.random.seed(42)
        sample_indices = np.random.choice(X.shape[1], min(50, X.shape[1]), replace=False)
        corr_matrix = np.corrcoef(X[:, sample_indices].T)
        
        im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_title('Correlation Matrix (50 Random Features)')
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Features')
        plt.colorbar(im, ax=axes[1, 0], label='Correlation')
        
        # Plot 5: Box plot of random features
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[1], min(10, X.shape[1]), replace=False)
        
        box_data = []
        box_labels = []
        for idx in random_indices:
            box_data.append(X[:, idx])
            if feature_names is not None:
                box_labels.append(feature_names[idx][:20])  # Truncate long names
            else:
                box_labels.append(f'F{idx}')
        
        bp = axes[1, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1, 1].set_title('Box Plot of Random Features')
        axes[1, 1].set_ylabel('Feature Values')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Feature statistics
        feature_stats = pd.DataFrame({
            'Mean': np.mean(X, axis=0),
            'Std': np.std(X, axis=0),
            'Min': np.min(X, axis=0),
            'Max': np.max(X, axis=0)
        })
        
        axes[1, 2].scatter(feature_stats['Mean'], feature_stats['Std'], 
                          alpha=0.5, s=10, c='purple')
        axes[1, 2].set_xlabel('Mean')
        axes[1, 2].set_ylabel('Standard Deviation')
        axes[1, 2].set_title('Mean vs Standard Deviation')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add density contours
        from scipy.stats import gaussian_kde
        xy = np.vstack([feature_stats['Mean'], feature_stats['Std']])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        axes[1, 2].scatter(feature_stats['Mean'].values[idx], 
                          feature_stats['Std'].values[idx], 
                          c=z[idx], s=10, cmap='viridis', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_feature_selection_comparison(self, results_df: pd.DataFrame,
                                         save_path: str = None) -> plt.Figure:
        """
        Plot feature selection method comparison results.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results dataframe with columns: method, dataset, accuracy, etc.
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Generated figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Feature Selection Method Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy by method and dataset
        if 'dataset' in results_df.columns and 'method' in results_df.columns:
            pivot_table = results_df.pivot_table(
                index='dataset',
                columns='method',
                values='accuracy',
                aggfunc='mean'
            )
            
            pivot_table.plot(kind='bar', ax=axes[0, 0], alpha=0.7)
            axes[0, 0].set_xlabel('Dataset')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Accuracy by Method and Dataset')
            axes[0, 0].legend(title='Method')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Method comparison box plot
        if 'method' in results_df.columns and 'accuracy' in results_df.columns:
            methods = results_df['method'].unique()
            box_data = []
            
            for method in methods:
                method_data = results_df[results_df['method'] == method]['accuracy']
                box_data.append(method_data)
            
            bp = axes[0, 1].boxplot(box_data, labels=methods, patch_artist=True)
            
            # Color the boxes
            colors = sns.color_palette("husl", len(methods))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[0, 1].set_xlabel('Method')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Method Performance Distribution')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # Add mean points
            means = [np.mean(data) for data in box_data]
            axes[0, 1].scatter(range(1, len(means) + 1), means, 
                              color='red', s=100, marker='D', label='Mean')
            axes[0, 1].legend()
        
        # Plot 3: Heatmap of best methods
        if 'dataset' in results_df.columns and 'method' in results_df.columns:
            # Find best method for each dataset
            best_methods = []
            for dataset in results_df['dataset'].unique():
                dataset_data = results_df[results_df['dataset'] == dataset]
                if not dataset_data.empty:
                    best_idx = dataset_data['accuracy'].idxmax()
                    best_method = dataset_data.loc[best_idx, 'method']
                    best_accuracy = dataset_data.loc[best_idx, 'accuracy']
                    best_methods.append({
                        'dataset': dataset,
                        'best_method': best_method,
                        'accuracy': best_accuracy
                    })
            
            best_df = pd.DataFrame(best_methods)
            
            # Create heatmap data
            methods = results_df['method'].unique()
            heatmap_data = pd.DataFrame(0, index=methods, columns=best_df['dataset'].unique())
            
            for _, row in best_df.iterrows():
                heatmap_data.loc[row['best_method'], row['dataset']] = row['accuracy']
            
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                       cbar_kws={'label': 'Accuracy'}, ax=axes[1, 0])
            axes[1, 0].set_xlabel('Dataset')
            axes[1, 0].set_ylabel('Best Method')
            axes[1, 0].set_title('Best Method per Dataset')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Performance vs number of features
        if 'n_features' in results_df.columns and 'accuracy' in results_df.columns:
            unique_n_features = sorted(results_df['n_features'].unique())
            
            for method in results_df['method'].unique():
                method_data = results_df[results_df['method'] == method]
                if not method_data.empty:
                    mean_accuracies = []
                    std_accuracies = []
                    
                    for n in unique_n_features:
                        n_data = method_data[method_data['n_features'] == n]['accuracy']
                        if len(n_data) > 0:
                            mean_accuracies.append(np.mean(n_data))
                            std_accuracies.append(np.std(n_data))
                    
                    axes[1, 1].errorbar(unique_n_features[:len(mean_accuracies)], 
                                       mean_accuracies, yerr=std_accuracies,
                                       marker='o', label=method, capsize=5)
            
            axes[1, 1].set_xlabel('Number of Selected Features')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Performance vs Feature Count')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrices(self, confusion_matrices: Dict[str, np.ndarray],
                               class_names: List[str] = None,
                               save_path: str = None) -> plt.Figure:
        """
        Plot confusion matrices for multiple models.
        
        Parameters:
        -----------
        confusion_matrices : Dict
            Dictionary with model names as keys and confusion matrices as values
        class_names : List[str], optional
            Class names for labeling
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Generated figure
        """
        n_models = len(confusion_matrices)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        if n_models == 1:
            axes = np.array([axes])
        if n_rows == 1 and n_cols > 1:
            axes = axes.reshape(1, -1)
        elif n_rows > 1 and n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Flatten axes array for easier indexing
        axes_flat = axes.flatten()
        
        for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
            ax = axes_flat[idx]
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot confusion matrix
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
            
            # Add text annotations
            thresh = cm_normalized.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                           ha="center", va="center",
                           color="white" if cm_normalized[i, j] > thresh else "black",
                           fontsize=9)
            
            # Set labels
            if class_names is not None:
                tick_marks = np.arange(len(class_names))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(class_names)
                ax.set_yticklabels(class_names)
            
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            ax.set_title(f'{model_name}\nAccuracy: {np.trace(cm)/np.sum(cm):.3f}')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(self, roc_data: Dict[str, Dict], save_path: str = None) -> plt.Figure:
        """
        Plot ROC curves for multiple models.
        
        Parameters:
        -----------
        roc_data : Dict
            Dictionary with model names as keys and dict of fpr, tpr, auc as values
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Generated figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curves
        for model_name, data in roc_data.items():
            fpr = data.get('fpr', [])
            tpr = data.get('tpr', [])
            auc = data.get('auc', 0)
            
            if len(fpr) > 0 and len(tpr) > 0:
                ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                               top_n: int = 20,
                               save_path: str = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Parameters:
        -----------
        importance_df : pd.DataFrame
            DataFrame with feature importance
        top_n : int
            Number of top features to plot
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Generated figure
        """
        # Sort by importance
        importance_sorted = importance_df.sort_values('importance', ascending=True).tail(top_n)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot 1: Horizontal bar chart
        bars = ax1.barh(range(len(importance_sorted)), 
                       importance_sorted['importance'].values,
                       alpha=0.7, color='steelblue')
        
        ax1.set_yticks(range(len(importance_sorted)))
        ax1.set_yticklabels(importance_sorted['feature'].values)
        ax1.set_xlabel('Importance')
        ax1.set_title(f'Top {top_n} Most Important Features')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, importance_sorted['importance'].values)):
            ax1.text(value + value * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.4f}', va='center', fontsize=9)
        
        # Plot 2: Cumulative importance
        cumulative_importance = importance_sorted['importance'].cumsum()
        total_importance = importance_sorted['importance'].sum()
        
        ax2.plot(range(1, len(cumulative_importance) + 1),
                cumulative_importance / total_importance * 100,
                marker='o', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance (%)')
        ax2.set_title('Cumulative Feature Importance')
        ax2.grid(True, alpha=0.3)
        
        # Add annotations for key points
        for n in [5, 10, 15]:
            if n <= len(cumulative_importance):
                cum_pct = cumulative_importance.iloc[n-1] / total_importance * 100
                ax2.annotate(f'{n} features\n{cum_pct:.1f}%',
                           (n, cum_pct), xytext=(10, 10),
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def create_interactive_dashboard(self, results: Dict, save_path: str = None):
        """
        Create interactive dashboard using Plotly.
        
        Parameters:
        -----------
        results : Dict
            Results dictionary
        save_path : str, optional
            Path to save HTML file
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Comparison', 'Feature Importance',
                           'ROC Curves', 'Confusion Matrix'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                  [{'type': 'scatter'}, {'type': 'heatmap'}]]
        )
        
        # Plot 1: Accuracy comparison
        if 'model_comparison' in results:
            models = list(results['model_comparison'].keys())
            accuracies = [results['model_comparison'][m].get('accuracy', 0) for m in models]
            
            fig.add_trace(
                go.Bar(x=models, y=accuracies, name='Accuracy',
                      marker_color='lightblue'),
                row=1, col=1
            )
        
        # Plot 2: Feature importance
        if 'feature_importance' in results:
            importance_df = results['feature_importance']
            top_features = importance_df.head(10)
            
            fig.add_trace(
                go.Bar(x=top_features['importance'], y=top_features['feature'],
                      orientation='h', name='Importance',
                      marker_color='lightgreen'),
                row=1, col=2
            )
        
        # Plot 3: ROC curve
        if 'roc_curves' in results:
            for model_name, roc_data in results['roc_curves'].items():
                fig.add_trace(
                    go.Scatter(x=roc_data['fpr'], y=roc_data['tpr'],
                              mode='lines', name=f'{model_name} (AUC={roc_data["auc"]:.3f})'),
                    row=2, col=1
                )
            
            # Add diagonal line
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                          name='Random', line=dict(dash='dash', color='gray')),
                row=2, col=1
            )
        
        # Plot 4: Confusion matrix
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            fig.add_trace(
                go.Heatmap(z=cm, colorscale='Blues', showscale=True),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Microarray Classification Dashboard",
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Importance", row=1, col=2)
        fig.update_xaxes(title_text="False Positive Rate", row=2, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        return fig
    
    def save_all_plots(self, results: Dict, output_dir: str = 'results/plots'):
        """
        Save all plots to directory.
        
        Parameters:
        -----------
        results : Dict
            Results dictionary
        output_dir : str
            Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving all plots to {output_dir}")
        
        # Save dataset summary if available
        if 'dataset_info' in results:
            self.plot_dataset_summary(
                results['dataset_info'],
                save_path=f'{output_dir}/dataset_summary.png'
            )
        
        # Save feature selection comparison if available
        if 'fs_comparison' in results:
            self.plot_feature_selection_comparison(
                results['fs_comparison'],
                save_path=f'{output_dir}/feature_selection_comparison.png'
            )
        
        # Save confusion matrices if available
        if 'confusion_matrices' in results:
            self.plot_confusion_matrices(
                results['confusion_matrices'],
                save_path=f'{output_dir}/confusion_matrices.png'
            )
        
        # Save ROC curves if available
        if 'roc_curves' in results:
            self.plot_roc_curves(
                results['roc_curves'],
                save_path=f'{output_dir}/roc_curves.png'
            )
        
        # Save feature importance if available
        if 'feature_importance' in results:
            self.plot_feature_importance(
                results['feature_importance'],
                save_path=f'{output_dir}/feature_importance.png'
            )
        
        # Create interactive dashboard
        self.create_interactive_dashboard(
            results,
            save_path=f'{output_dir}/interactive_dashboard.html'
        )
        
        logger.info(f"All plots saved to {output_dir}")


# Standalone visualization functions
def plot_correlation_matrix(X: np.ndarray, feature_names: List[str] = None,
                           max_features: int = 50, save_path: str = None) -> plt.Figure:
    """Plot correlation matrix for features."""
    # Limit number of features for visualization
    n_features = min(X.shape[1], max_features)
    if feature_names is not None:
        feature_names = feature_names[:n_features]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X[:, :n_features].T)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Set labels
    if feature_names is not None:
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=90)
        ax.set_yticklabels(feature_names)
    
    ax.set_title(f'Feature Correlation Matrix (Top {n_features} Features)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def plot_learning_curve(train_sizes, train_scores, val_scores, 
                       model_name: str = None, save_path: str = None) -> plt.Figure:
    """Plot learning curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                   alpha=0.1, color='blue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='green', label='Validation score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                   alpha=0.1, color='green')
    
    # Customize plot
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score')
    ax.set_title(f'Learning Curve{" - " + model_name if model_name else ""}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(precision, recall, average_precision,
                               model_name: str = None, save_path: str = None) -> plt.Figure:
    """Plot precision-recall curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(recall, precision, lw=2, 
           label=f'AP = {average_precision:.3f}')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve{" - " + model_name if model_name else ""}')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


# Example usage function
def example_usage():
    """Example usage of visualization utilities."""
    import numpy as np
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 1000
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    feature_names = [f'Gene_{i}' for i in range(n_features)]
    
    # Create sample results
    dataset_info = {
        'Leukemia': {'n_samples': 72, 'n_features': 7129, 'missing_values': 100, 'sparsity': 10},
        'SRBCT': {'n_samples': 83, 'n_features': 2308, 'missing_values': 50, 'sparsity': 5},
        'Lymphoma': {'n_samples': 96, 'n_features': 4026, 'missing_values': 200, 'sparsity': 15}
    }
    
    results_df = pd.DataFrame({
        'method': ['MIM', 'JMI', 'MRMR', 'MIM', 'JMI', 'MRMR'],
        'dataset': ['Leukemia', 'Leukemia', 'Leukemia', 'SRBCT', 'SRBCT', 'SRBCT'],
        'accuracy': [0.85, 0.91, 0.87, 0.92, 0.97, 0.94],
        'n_features': [100, 100, 100, 100, 100, 100]
    })
    
    confusion_matrices = {
        'Random Forest': np.array([[45, 5], [8, 42]]),
        'SVM': np.array([[43, 7], [6, 44]])
    }
    
    roc_data = {
        'Random Forest': {'fpr': [0, 0.1, 0.2, 0.5, 1], 
                         'tpr': [0, 0.3, 0.6, 0.9, 1], 
                         'auc': 0.92},
        'SVM': {'fpr': [0, 0.2, 0.4, 0.7, 1], 
               'tpr': [0, 0.4, 0.7, 0.95, 1], 
               'auc': 0.88}
    }
    
    importance_df = pd.DataFrame({
        'feature': feature_names[:20],
        'importance': np.random.rand(20)
    }).sort_values('importance', ascending=False)
    
    # Initialize visualizer
    visualizer = MicroarrayVisualizer()
    
    # Create plots
    visualizer.plot_dataset_summary(dataset_info, save_path='dataset_summary.png')
    visualizer.plot_feature_distribution(X, y, feature_names, save_path='feature_distribution.png')
    visualizer.plot_feature_selection_comparison(results_df, save_path='fs_comparison.png')
    visualizer.plot_confusion_matrices(confusion_matrices, save_path='confusion_matrices.png')
    visualizer.plot_roc_curves(roc_data, save_path='roc_curves.png')
    visualizer.plot_feature_importance(importance_df, save_path='feature_importance.png')
    
    # Create interactive dashboard
    results = {
        'model_comparison': {
            'Random Forest': {'accuracy': 0.91},
            'SVM': {'accuracy': 0.88},
            'XGBoost': {'accuracy': 0.89}
        },
        'feature_importance': importance_df,
        'roc_curves': roc_data,
        'confusion_matrix': confusion_matrices['Random Forest']
    }
    
    visualizer.create_interactive_dashboard(results, save_path='dashboard.html')
    
    print("Example visualizations created successfully!")
    
    return visualizer


if __name__ == "__main__":
    # Run example
    visualizer = example_usage()