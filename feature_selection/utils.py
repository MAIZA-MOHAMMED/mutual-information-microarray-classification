"""
Utility functions for mutual information estimation and feature selection.
"""

import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.special import digamma

def estimate_mutual_information(x, y, discrete_x=False, discrete_y=False, 
                               method='histogram', bins=10, k=3):
    """
    Estimate mutual information between two variables.
    
    Parameters:
    -----------
    x, y : array-like
        Input variables
    discrete_x, discrete_y : bool
        Whether variables are discrete
    method : str
        Estimation method: 'histogram', 'kde', or 'knn'
    bins : int
        Number of bins for histogram method
    k : int
        Number of neighbors for kNN method
        
    Returns:
    --------
    mi : float
        Estimated mutual information
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    n_samples = len(x)
    
    if method == 'histogram':
        # Histogram-based estimation
        if discrete_x and discrete_y:
            # Both discrete: use contingency table
            contingency = np.histogram2d(x, y, 
                                        bins=(len(np.unique(x)), len(np.unique(y))))[0]
        else:
            # At least one continuous: use bins
            contingency = np.histogram2d(x, y, bins=bins)[0]
        
        # Convert to probabilities
        p_xy = contingency / n_samples
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        
        # Calculate MI
        mi = 0
        for i in range(p_xy.shape[0]):
            for j in range(p_xy.shape[1]):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return mi
    
    elif method == 'kde':
        # Kernel Density Estimation based MI
        # Estimate marginal and joint densities
        if discrete_x and discrete_y:
            # For discrete variables, use histogram
            return estimate_mutual_information(x, y, method='histogram')
        
        # Bandwidth selection using Silverman's rule
        bw_x = 1.06 * np.std(x) * (n_samples ** (-0.2)) if not discrete_x else None
        bw_y = 1.06 * np.std(y) * (n_samples ** (-0.2)) if not discrete_y else None
        
        # For continuous variables, use KDE
        mi = 0
        for i in range(n_samples):
            # This is a simplified estimation
            # In practice, use specialized MI estimation libraries
            pass
        
        # Fallback to histogram for now
        return estimate_mutual_information(x, y, method='histogram', bins=bins)
    
    elif method == 'knn':
        # k-Nearest Neighbors based MI estimation (Kraskov et al., 2004)
        if discrete_x or discrete_y:
            # kNN works best for continuous variables
            return estimate_mutual_information(x, y, method='histogram')
        
        # Combine x and y for kNN search
        xy = np.column_stack([x.reshape(-1, 1), y.reshape(-1, 1)])
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(xy)
        distances, _ = nbrs.kneighbors(xy)
        
        # Maximum distance to k-th neighbor
        epsilon = distances[:, -1]
        
        # Count neighbors within epsilon in each marginal space
        n_x = np.zeros(n_samples)
        n_y = np.zeros(n_samples)
        
        for i in range(n_samples):
            n_x[i] = np.sum(np.abs(x - x[i]) <= epsilon[i])
            n_y[i] = np.sum(np.abs(y - y[i]) <= epsilon[i])
        
        # Kraskov estimator
        mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n_samples)
        
        return max(mi, 0)  # MI is non-negative
    
    else:
        raise ValueError(f"Unknown method: {method}")


def mutual_info_matrix(X, discrete_features=False, method='histogram', **kwargs):
    """
    Calculate mutual information matrix between all pairs of features.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    discrete_features : bool or array-like
        Whether features are discrete
    method : str
        MI estimation method
    **kwargs : dict
        Additional arguments for MI estimation
        
    Returns:
    --------
    mi_matrix : array, shape (n_features, n_features)
        Mutual information matrix (symmetric)
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape
    
    # Handle discrete_features parameter
    if isinstance(discrete_features, bool):
        discrete_features = [discrete_features] * n_features
    elif len(discrete_features) != n_features:
        raise ValueError("discrete_features must match number of features")
    
    mi_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i, n_features):
            if i == j:
                # Self-information (entropy)
                mi_matrix[i, j] = estimate_entropy(
                    X[:, i], 
                    discrete=discrete_features[i],
                    method=method,
                    **kwargs
                )
            else:
                mi_matrix[i, j] = estimate_mutual_information(
                    X[:, i], X[:, j],
                    discrete_x=discrete_features[i],
                    discrete_y=discrete_features[j],
                    method=method,
                    **kwargs
                )
                mi_matrix[j, i] = mi_matrix[i, j]
    
    return mi_matrix


def estimate_entropy(x, discrete=False, method='histogram', bins=10):
    """
    Estimate entropy of a variable.
    
    Parameters:
    -----------
    x : array-like
        Input variable
    discrete : bool
        Whether variable is discrete
    method : str
        Estimation method
    bins : int
        Number of bins for histogram method
        
    Returns:
    --------
    entropy : float
        Estimated entropy
    """
    x = np.asarray(x).flatten()
    
    if method == 'histogram':
        if discrete:
            # For discrete variables, use empirical distribution
            values, counts = np.unique(x, return_counts=True)
            probs = counts / len(x)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            # For continuous variables, use histogram
            hist, _ = np.histogram(x, bins=bins, density=True)
            probs = hist / np.sum(hist)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            # Add correction for bin width
            bin_width = (np.max(x) - np.min(x)) / bins
            entropy += np.log(bin_width)
        
        return entropy
    
    elif method == 'knn':
        # kNN entropy estimation
        from sklearn.neighbors import NearestNeighbors
        
        x_reshaped = x.reshape(-1, 1)
        n_samples = len(x)
        k = min(3, n_samples - 1)
        
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(x_reshaped)
        distances, _ = nbrs.kneighbors(x_reshaped)
        
        # Maximum distance to k-th neighbor
        epsilon = distances[:, -1]
        
        # Volume of d-dimensional ball
        d = 1  # dimension
        volume_unit_ball = np.pi**(d/2) / np.math.gamma(d/2 + 1)
        
        # Entropy estimation
        entropy = np.mean(np.log(epsilon + 1e-10)) + np.log(volume_unit_ball) 
        entropy += digamma(n_samples) - digamma(k)
        
        return entropy
    
    else:
        raise ValueError(f"Unknown method: {method}")


def normalize_mi_scores(scores, method='minmax'):
    """
    Normalize mutual information scores.
    
    Parameters:
    -----------
    scores : array-like
        MI scores to normalize
    method : str
        Normalization method: 'minmax', 'zscore', or 'softmax'
        
    Returns:
    --------
    normalized_scores : array
        Normalized scores
    """
    scores = np.asarray(scores)
    
    if method == 'minmax':
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val > min_val:
            return (scores - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(scores)
    
    elif method == 'zscore':
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        if std_val > 0:
            return (scores - mean_val) / std_val
        else:
            return np.zeros_like(scores)
    
    elif method == 'softmax':
        # Softmax normalization
        exp_scores = np.exp(scores - np.max(scores))  # For numerical stability
        return exp_scores / np.sum(exp_scores)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def select_top_features(scores, n_features, threshold=None):
    """
    Select top features based on scores.
    
    Parameters:
    -----------
    scores : array-like
        Feature scores
    n_features : int
        Number of features to select
    threshold : float or None
        Minimum score threshold
        
    Returns:
    --------
    selected_indices : array
        Indices of selected features
    """
    scores = np.asarray(scores)
    
    if threshold is not None:
        # Select features above threshold
        above_threshold = scores >= threshold
        selected = np.where(above_threshold)[0]
        
        # Sort by score (descending)
        sorted_indices = selected[np.argsort(-scores[selected])]
        
        return sorted_indices[:n_features] if len(sorted_indices) > n_features else sorted_indices
    
    else:
        # Select top n_features
        if n_features > len(scores):
            n_features = len(scores)
        
        # Get indices of top scores
        top_indices = np.argsort(-scores)[:n_features]
        
        return top_indices


def calculate_redundancy(feature_set, mi_matrix):
    """
    Calculate average redundancy among a set of features.
    
    Parameters:
    -----------
    feature_set : array-like
        Indices of features
    mi_matrix : array-like
        Mutual information matrix
        
    Returns:
    --------
    redundancy : float
        Average pairwise MI among features
    """
    if len(feature_set) < 2:
        return 0.0
    
    total_mi = 0
    count = 0
    
    for i in range(len(feature_set)):
        for j in range(i + 1, len(feature_set)):
            f1 = feature_set[i]
            f2 = feature_set[j]
            total_mi += mi_matrix[f1, f2]
            count += 1
    
    return total_mi / count if count > 0 else 0.0


def calculate_relevance(feature_set, mi_with_target):
    """
    Calculate average relevance of features to target.
    
    Parameters:
    -----------
    feature_set : array-like
        Indices of features
    mi_with_target : array-like
        MI scores with target
        
    Returns:
    --------
    relevance : float
        Average MI with target
    """
    if len(feature_set) == 0:
        return 0.0
    
    total_mi = np.sum(mi_with_target[feature_set])
    return total_mi / len(feature_set)


def jmi_criterion(candidate, selected, mi_with_target, mi_matrix, alpha=1.0):
    """
    Calculate JMI criterion for a candidate feature.
    
    Parameters:
    -----------
    candidate : int
        Candidate feature index
    selected : list
        Selected feature indices
    mi_with_target : array
        MI scores with target
    mi_matrix : array
        Mutual information matrix
    alpha : float
        Weight parameter
        
    Returns:
    --------
    score : float
        JMI score
    """
    relevance = mi_with_target[candidate]
    
    if not selected:
        return relevance
    
    redundancy = 0
    conditional_info = 0
    
    for s in selected:
        redundancy += mi_matrix[candidate, s]
        # Approximate conditional mutual information
        # I(candidate; s|y) ≈ I(candidate; s) - I(candidate; y) - I(s; y) + I(candidate, s; y)
        # Simplified version used in practice
        conditional_info += max(0, mi_matrix[candidate, s] - mi_with_target[candidate] - mi_with_target[s])
    
    redundancy /= len(selected)
    conditional_info /= len(selected)
    
    # JMI score
    score = relevance - alpha * redundancy + (1 - alpha) * conditional_info
    
    return score


def mrmr_criterion(candidate, selected, mi_with_target, mi_matrix, beta=1.0):
    """
    Calculate MRMR criterion for a candidate feature.
    
    Parameters:
    -----------
    candidate : int
        Candidate feature index
    selected : list
        Selected feature indices
    mi_with_target : array
        MI scores with target
    mi_matrix : array
        Mutual information matrix
    beta : float
        Weight parameter
        
    Returns:
    --------
    score : float
        MRMR score
    """
    relevance = mi_with_target[candidate]
    
    if not selected:
        return relevance
    
    redundancy = 0
    for s in selected:
        redundancy += mi_matrix[candidate, s]
    
    redundancy /= len(selected)
    
    # MRMR score
    score = relevance - beta * redundancy
    
    return score


def feature_selection_summary(selected_features, feature_names=None, 
                             mi_scores=None, method_name="Feature Selection"):
    """
    Create a summary of feature selection results.
    
    Parameters:
    -----------
    selected_features : array-like
        Indices of selected features
    feature_names : list or None
        Names of features
    mi_scores : array-like or None
        MI scores for features
    method_name : str
        Name of feature selection method
        
    Returns:
    --------
    summary : str
        Text summary
    """
    summary = f"{method_name} Results\n"
    summary += "=" * 50 + "\n"
    summary += f"Selected {len(selected_features)} features\n\n"
    
    if feature_names is not None and mi_scores is not None:
        summary += "Top 20 selected features:\n"
        summary += "-" * 50 + "\n"
        
        for i, idx in enumerate(selected_features[:20], 1):
            name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            score = mi_scores[idx] if idx < len(mi_scores) else "N/A"
            summary += f"{i:3d}. {name:30s} | MI Score: {score:.6f}\n"
    
    elif mi_scores is not None:
        summary += "Selected features with scores:\n"
        summary += "-" * 50 + "\n"
        
        for i, idx in enumerate(selected_features[:20], 1):
            score = mi_scores[idx] if idx < len(mi_scores) else "N/A"
            summary += f"{i:3d}. Feature {idx:5d} | MI Score: {score:.6f}\n"
    
    return summary