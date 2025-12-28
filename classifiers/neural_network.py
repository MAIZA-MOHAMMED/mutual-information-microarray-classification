"""
Neural Network classifier optimized for microarray data.

Based on the paper's hyperparameter tuning:
- Layers: 1-3 (optimal: 2)
- Neurons per layer: 32-128 (optimal: 64)
- Dropout rate: 0.2-0.5 (optimal: 0.3)
- Learning rate: 0.001-0.01 (optimal: 0.001)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks, optimizers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. Neural network functionality will be limited.")

# Scikit-learn imports for compatibility
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MicroarrayNeuralNetwork(BaseEstimator, ClassifierMixin):
    """
    Neural Network classifier optimized for high-dimensional microarray data.
    
    Architecture and hyperparameters based on paper optimization:
    - 1-3 hidden layers (optimal: 2)
    - 32-128 neurons per layer (optimal: 64)
    - Dropout rate: 0.2-0.5 (optimal: 0.3)
    - Learning rate: 0.001-0.01 (optimal: 0.001)
    """
    
    def __init__(self, 
                 n_features: int,
                 n_classes: int = 2,
                 n_layers: int = 2,
                 neurons_per_layer: int = 64,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 batch_size: int = 16,
                 epochs: int = 200,
                 validation_split: float = 0.2,
                 early_stopping_patience: int = 20,
                 l2_regularization: float = 0.01,
                 use_batch_norm: bool = True,
                 random_state: int = 42,
                 verbose: int = 1):
        """
        Initialize neural network classifier.
        
        Parameters:
        -----------
        n_features : int
            Number of input features (genes)
        n_classes : int
            Number of output classes (default: 2 for binary classification)
        n_layers : int
            Number of hidden layers (1-3, optimal: 2 as per paper)
        neurons_per_layer : int
            Number of neurons in each hidden layer (32-128, optimal: 64)
        dropout_rate : float
            Dropout rate for regularization (0.2-0.5, optimal: 0.3)
        learning_rate : float
            Learning rate for Adam optimizer (0.001-0.01, optimal: 0.001)
        batch_size : int
            Batch size for training (default: 16)
        epochs : int
            Maximum number of training epochs (default: 200)
        validation_split : float
            Fraction of training data to use for validation (default: 0.2)
        early_stopping_patience : int
            Patience for early stopping (default: 20)
        l2_regularization : float
            L2 regularization strength (default: 0.01)
        use_batch_norm : bool
            Whether to use batch normalization (default: True)
        random_state : int
            Random seed for reproducibility
        verbose : int
            Verbosity level (0: silent, 1: progress bars, 2: one line per epoch)
        
        Raises:
        -------
        ImportError: If TensorFlow is not available
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for NeuralNetworkClassifier. "
                "Install with: pip install tensorflow"
            )
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        # Store parameters
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.neurons_per_layer = neurons_per_layer
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.l2_regularization = l2_regularization
        self.use_batch_norm = use_batch_norm
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize model components
        self.model = None
        self.history = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.best_epoch = 0
        self.feature_importances_ = None
        
        # Performance tracking
        self.training_time = None
        self.inference_time = None
        
        logger.info(f"Initialized Neural Network with {n_layers} layers, "
                   f"{neurons_per_layer} neurons/layer, dropout={dropout_rate}")
    
    def build_model(self) -> keras.Model:
        """
        Build the neural network architecture.
        
        Returns:
        --------
        model : keras.Model
            Compiled neural network model
        """
        logger.info(f"Building neural network with {self.n_layers} hidden layers")
        
        model = Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.n_features,)))
        
        # Hidden layers with regularization
        for i in range(self.n_layers):
            # Dense layer with L2 regularization
            model.add(layers.Dense(
                self.neurons_per_layer,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_regularization),
                name=f'hidden_{i+1}'
            ))
            
            # Batch normalization (optional)
            if self.use_batch_norm:
                model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
            
            # Dropout for regularization
            model.add(layers.Dropout(
                self.dropout_rate,
                name=f'dropout_{i+1}'
            ))
        
        # Output layer
        if self.n_classes == 2:
            # Binary classification
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
            output_activation = 'sigmoid'
            metrics = ['accuracy', keras.metrics.AUC(name='auc')]
        else:
            # Multi-class classification
            model.add(layers.Dense(
                self.n_classes,
                activation='softmax',
                name='output'
            ))
            loss = 'categorical_crossentropy'
            output_activation = 'softmax'
            metrics = ['accuracy']
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        # Store model architecture info
        self.model_info = {
            'n_layers': self.n_layers,
            'neurons_per_layer': self.neurons_per_layer,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'l2_regularization': self.l2_regularization,
            'output_activation': output_activation,
            'total_params': model.count_params()
        }
        
        logger.info(f"Model built with {self.model_info['total_params']:,} parameters")
        
        return model
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray = None, 
                       fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess microarray data for neural network.
        
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
        X_scaled : np.ndarray
            Scaled feature matrix
        y_encoded : np.ndarray or None
            Encoded target labels (if y provided)
        """
        # Scale features (important for neural networks)
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Encode labels if provided
        y_encoded = None
        if y is not None:
            if fit:
                y_encoded = self.label_encoder.fit_transform(y)
                self.classes_ = self.label_encoder.classes_
            else:
                y_encoded = self.label_encoder.transform(y)
        
        return X_scaled, y_encoded
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'MicroarrayNeuralNetwork':
        """
        Train the neural network.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
        X_val, y_val : np.ndarray, optional
            Validation data (if not provided, use validation_split)
            
        Returns:
        --------
        self : MicroarrayNeuralNetwork
            Trained classifier
        """
        import time
        start_time = time.time()
        
        logger.info(f"Training neural network on {X.shape[0]} samples, {X.shape[1]} features")
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_processed, y_val_processed = self.preprocess_data(X_val, y_val, fit=False)
            
            # Convert to one-hot for multi-class
            if self.n_classes > 2:
                y_val_processed = to_categorical(y_val_processed, self.n_classes)
                validation_data = (X_val_processed, y_val_processed)
            else:
                validation_data = (X_val_processed, y_val_processed)
            logger.info(f"Using provided validation data: {X_val.shape[0]} samples")
        
        # Convert labels for multi-class classification
        if self.n_classes > 2:
            y_processed = to_categorical(y_processed, self.n_classes)
        
        # Define callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=self.verbose
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=self.early_stopping_patience // 2,
                min_lr=1e-6,
                verbose=self.verbose
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_processed,
            y_processed,
            validation_data=validation_data,
            validation_split=self.validation_split if validation_data is None else 0.0,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks_list,
            verbose=self.verbose,
            shuffle=True
        )
        
        # Find best epoch
        if validation_data is not None:
            val_loss = self.history.history['val_loss']
        else:
            val_loss = self.history.history['loss']
        
        self.best_epoch = np.argmin(val_loss) + 1
        
        # Calculate training time
        self.training_time = time.time() - start_time
        
        # Calculate feature importances
        self._calculate_feature_importance(X_processed, y_processed)
        
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        logger.info(f"Best epoch: {self.best_epoch}/{self.epochs}")
        logger.info(f"Final loss: {self.history.history['loss'][-1]:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted class labels
        """
        import time
        start_time = time.time()
        
        # Preprocess features
        X_processed, _ = self.preprocess_data(X, fit=False)
        
        # Make predictions
        predictions = self.model.predict(X_processed, verbose=0)
        
        # Convert to class labels
        if self.n_classes == 2:
            # Binary classification
            y_pred = (predictions > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            y_pred = np.argmax(predictions, axis=1)
        
        # Decode labels
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        
        # Calculate inference time
        self.inference_time = time.time() - start_time
        
        return y_pred_decoded
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
            
        Returns:
        --------
        probabilities : np.ndarray
            Class probabilities
        """
        # Preprocess features
        X_processed, _ = self.preprocess_data(X, fit=False)
        
        # Predict probabilities
        probabilities = self.model.predict(X_processed, verbose=0)
        
        # For binary classification, reshape to (n_samples, 2)
        if self.n_classes == 2:
            prob_positive = probabilities.flatten()
            prob_negative = 1 - prob_positive
            probabilities = np.column_stack([prob_negative, prob_positive])
        
        return probabilities
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X : np.ndarray
            Test features
        y : np.ndarray
            True labels
            
        Returns:
        --------
        metrics : Dict
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
        }
        
        # Add AUC if applicable
        if self.n_classes == 2:
            try:
                metrics['auc'] = roc_auc_score(y, y_pred_proba[:, 1])
            except:
                metrics['auc'] = np.nan
        
        # Add confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Per-class metrics for reasonable number of classes
        if len(np.unique(y)) <= 10:
            report = classification_report(y, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
        
        logger.info(f"Evaluation: Accuracy = {metrics['accuracy']:.4f}, "
                   f"F1 = {metrics['f1']:.4f}")
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                      n_splits: int = 5,
                      feature_selector = None,
                      random_state: int = None) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        n_splits : int
            Number of cross-validation folds
        feature_selector : object, optional
            Feature selector with fit/transform methods
        random_state : int, optional
            Random seed for cross-validation
            
        Returns:
        --------
        cv_results : Dict
            Cross-validation results
        """
        if random_state is None:
            random_state = self.random_state
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        cv_results = {
            'fold_accuracies': [],
            'fold_precisions': [],
            'fold_recalls': [],
            'fold_f1s': [],
            'fold_training_times': [],
            'fold_models': [],
            'selected_features': []
        }
        
        logger.info(f"Starting {n_splits}-fold cross-validation")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            logger.info(f"Processing fold {fold}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Apply feature selection if provided
            if feature_selector is not None:
                # Fit on training data only
                feature_selector.fit(X_train, y_train)
                X_train_selected = feature_selector.transform(X_train)
                X_val_selected = feature_selector.transform(X_val)
                cv_results['selected_features'].append(feature_selector.selected_features)
            else:
                X_train_selected = X_train
                X_val_selected = X_val
            
            # Update n_features for this fold
            self.n_features = X_train_selected.shape[1]
            
            # Create fresh model for this fold
            fold_model = MicroarrayNeuralNetwork(
                n_features=self.n_features,
                n_classes=self.n_classes,
                n_layers=self.n_layers,
                neurons_per_layer=self.neurons_per_layer,
                dropout_rate=self.dropout_rate,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_split,
                early_stopping_patience=self.early_stopping_patience,
                l2_regularization=self.l2_regularization,
                use_batch_norm=self.use_batch_norm,
                random_state=self.random_state + fold,
                verbose=0
            )
            
            # Train model
            fold_model.fit(X_train_selected, y_train, X_val_selected, y_val)
            
            # Evaluate
            fold_metrics = fold_model.evaluate(X_val_selected, y_val)
            
            # Store results
            cv_results['fold_accuracies'].append(fold_metrics['accuracy'])
            cv_results['fold_precisions'].append(fold_metrics['precision'])
            cv_results['fold_recalls'].append(fold_metrics['recall'])
            cv_results['fold_f1s'].append(fold_metrics['f1'])
            cv_results['fold_training_times'].append(fold_model.training_time)
            cv_results['fold_models'].append(fold_model)
            
            logger.info(f"Fold {fold}: Accuracy = {fold_metrics['accuracy']:.4f}, "
                       f"F1 = {fold_metrics['f1']:.4f}")
        
        # Calculate summary statistics
        cv_results['mean_accuracy'] = np.mean(cv_results['fold_accuracies'])
        cv_results['std_accuracy'] = np.std(cv_results['fold_accuracies'])
        cv_results['mean_f1'] = np.mean(cv_results['fold_f1s'])
        cv_results['std_f1'] = np.std(cv_results['fold_f1s'])
        cv_results['mean_training_time'] = np.mean(cv_results['fold_training_times'])
        
        logger.info(f"\nCross-validation results:")
        logger.info(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        logger.info(f"Mean F1-score: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
        logger.info(f"Mean training time: {cv_results['mean_training_time']:.2f} seconds")
        
        return cv_results
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculate feature importance using gradient-based method.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        """
        try:
            # Use gradient-based importance
            with tf.GradientTape() as tape:
                # Get model predictions
                inputs = tf.constant(X, dtype=tf.float32)
                tape.watch(inputs)
                predictions = self.model(inputs)
                
                # For binary classification
                if self.n_classes == 2:
                    # Use mean squared error as loss
                    y_tensor = tf.constant(y.reshape(-1, 1), dtype=tf.float32)
                    loss = tf.reduce_mean(tf.square(predictions - y_tensor))
                else:
                    # For multi-class, use categorical cross-entropy
                    y_one_hot = tf.constant(to_categorical(y, self.n_classes), dtype=tf.float32)
                    loss = tf.keras.losses.categorical_crossentropy(y_one_hot, predictions)
                    loss = tf.reduce_mean(loss)
            
            # Compute gradients
            gradients = tape.gradient(loss, inputs)
            
            # Average absolute gradients across samples
            importance = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()
            
            # Normalize to sum to 1
            if np.sum(importance) > 0:
                importance = importance / np.sum(importance)
            
            self.feature_importances_ = importance
            
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {str(e)}")
            self.feature_importances_ = np.ones(self.n_features) / self.n_features
    
    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.
        
        Parameters:
        -----------
        feature_names : List[str], optional
            Names of features
            
        Returns:
        --------
        importance_df : pd.DataFrame
            DataFrame with feature importance
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature importance not calculated. Train the model first.")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Add cumulative importance
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        importance_df['importance_percentage'] = importance_df['importance'] * 100
        
        return importance_df
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model architecture and training summary.
        
        Returns:
        --------
        summary : Dict
            Model summary dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        summary = {
            'architecture': self.model_info,
            'training': {
                'best_epoch': self.best_epoch,
                'total_epochs': len(self.history.history['loss']),
                'training_time': self.training_time,
                'inference_time': self.inference_time
            },
            'performance': {}
        }
        
        # Add training history if available
        if self.history is not None:
            summary['history'] = {
                'loss': self.history.history['loss'],
                'accuracy': self.history.history['accuracy'] if 'accuracy' in self.history.history else None,
                'val_loss': self.history.history['val_loss'] if 'val_loss' in self.history.history else None,
                'val_accuracy': self.history.history['val_accuracy'] if 'val_accuracy' in self.history.history else None
            }
        
        return summary
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Save TensorFlow model
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save additional metadata
        metadata = {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'classes_': self.classes_.tolist() if self.classes_ is not None else None,
            'model_info': self.model_info,
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'label_encoder_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None
        }
        
        import json
        metadata_file = filepath.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {metadata_file}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MicroarrayNeuralNetwork':
        """
        Load model from file.
        
        Parameters:
        -----------
        filepath : str
            Path to saved model
            
        Returns:
        --------
        model : MicroarrayNeuralNetwork
            Loaded model
        """
        # Load TensorFlow model
        loaded_model = keras.models.load_model(filepath)
        
        # Load metadata
        metadata_file = filepath.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Create classifier instance
        classifier = cls(
            n_features=metadata['n_features'],
            n_classes=metadata['n_classes']
        )
        
        # Restore model and metadata
        classifier.model = loaded_model
        classifier.classes_ = np.array(metadata['classes_']) if metadata['classes_'] else None
        classifier.model_info = metadata['model_info']
        
        # Restore scaler
        if metadata['scaler_mean'] and metadata['scaler_scale']:
            classifier.scaler.mean_ = np.array(metadata['scaler_mean'])
            classifier.scaler.scale_ = np.array(metadata['scaler_scale'])
            classifier.scaler.n_samples_seen_ = len(metadata['scaler_mean'])
        
        # Restore label encoder
        if metadata['label_encoder_classes']:
            classifier.label_encoder.classes_ = np.array(metadata['label_encoder_classes'])
        
        logger.info(f"Model loaded from {filepath}")
        return classifier
    
    def plot_training_history(self, save_path: str = None) -> None:
        """
        Plot training history.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save plot
        """
        import matplotlib.pyplot as plt
        
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].axvline(x=self.best_epoch - 1, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        if 'accuracy' in self.history.history:
            axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in self.history.history:
                axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            axes[1].axvline(x=self.best_epoch - 1, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()


# Example usage
def example_usage():
    """Example usage of the neural network classifier."""
    import numpy as np
    
    # Generate synthetic microarray-like data
    np.random.seed(42)
    n_samples = 200
    n_features = 1000
    
    # Create informative features (10% of total)
    n_informative = 100
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Add signal to informative features
    for i in range(n_informative):
        X[y == 0, i] += 1.5  # Class 0 has higher expression
        X[y == 1, i] -= 1.5  # Class 1 has lower expression
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Initialize neural network
    nn = MicroarrayNeuralNetwork(
        n_features=n_features,
        n_layers=2,
        neurons_per_layer=64,
        dropout_rate=0.3,
        learning_rate=0.001,
        epochs=50,
        verbose=1
    )
    
    # Train model
    print("\nTraining neural network...")
    nn.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = nn.evaluate(X_test, y_test)
    
    print(f"\nTest Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    
    if 'auc' in metrics:
        print(f"AUC-ROC: {metrics['auc']:.4f}")
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_results = nn.cross_validate(X_train, y_train, n_splits=3, verbose=0)
    
    print(f"\nCross-validation results:")
    print(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"Mean F1-score: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
    
    # Get feature importance
    print("\nTop 10 most important features:")
    importance_df = nn.get_feature_importance()
    print(importance_df.head(10))
    
    # Get model summary
    summary = nn.get_model_summary()
    print(f"\nModel summary:")
    print(f"Architecture: {summary['architecture']['n_layers']} layers, "
          f"{summary['architecture']['neurons_per_layer']} neurons/layer")
    print(f"Total parameters: {summary['architecture']['total_params']:,}")
    print(f"Best epoch: {summary['training']['best_epoch']}")
    
    # Plot training history
    nn.plot_training_history()
    
    return nn, metrics


if __name__ == "__main__":
    # Run example
    nn, metrics = example_usage()
