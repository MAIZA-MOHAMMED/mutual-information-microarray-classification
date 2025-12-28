"""
Script to download and preprocess microarray datasets used in the paper.
All datasets are publicly available from original sources.
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import tarfile
import zipfile
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MicroarrayDatasetDownloader:
    """
    Downloader for microarray datasets used in the paper.
    """
    
    # Dataset URLs and information
    DATASETS = {
        'leukemia': {
            'url': 'https://web.stanford.edu/~hastie/CASI_files/DATA/leukemia.csv',
            'filename': 'leukemia.csv',
            'description': 'Leukemia dataset (Golub et al., 1999)',
            'n_features': 7129,
            'n_samples': 72
        },
        'colon_cancer': {
            'url': 'https://github.com/jundongl/scikit-feature/raw/master/skfeature/data/colon.mat',
            'filename': 'colon.mat',
            'description': 'Colon cancer dataset (Alon et al., 1999)',
            'n_features': 2000,
            'n_samples': 62
        },
        'srbct': {
            'url': 'https://github.com/jundongl/scikit-feature/raw/master/skfeature/data/SRBCT.mat',
            'filename': 'SRBCT.mat',
            'description': 'Small Round Blue Cell Tumors dataset (Khan et al., 2001)',
            'n_features': 2308,
            'n_samples': 83
        },
        'lymphoma': {
            'url': 'https://github.com/jundongl/scikit-feature/raw/master/skfeature/data/lymphoma.mat',
            'filename': 'lymphoma.mat',
            'description': 'Lymphoma dataset (Alizadeh et al., 2000)',
            'n_features': 4026,
            'n_samples': 96
        },
        'dlbcl': {
            'url': 'https://github.com/jundongl/scikit-feature/raw/master/skfeature/data/dlbcl.mat',
            'filename': 'dlbcl.mat',
            'description': 'Diffuse Large B-Cell Lymphoma dataset (Shipp et al., 2002)',
            'n_features': 3812,
            'n_samples': 42
        }
    }
    
    # Alternative sources for datasets
    ALTERNATIVE_SOURCES = {
        'brain_cancer': {
            'source': 'GEO GSE4412',
            'instructions': 'Please download from GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4412'
        },
        'prostate_tumor': {
            'source': 'GEO GSE6919',
            'instructions': 'Please download from GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE6919'
        },
        'lung_cancer': {
            'source': 'GEO GSE1987',
            'instructions': 'Please download from GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE1987'
        },
        '11_tumors': {
            'source': 'https://portals.broadinstitute.org/cgi-bin/cancer/datasets.cgi',
            'instructions': 'Please download from the Broad Institute website'
        }
    }
    
    def __init__(self, data_dir='data/raw'):
        """
        Initialize downloader.
        
        Parameters:
        -----------
        data_dir : str
            Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, dataset_name, force=False):
        """
        Download a specific dataset.
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to download
        force : bool
            Force re-download even if file exists
            
        Returns:
        --------
        filepath : Path
            Path to downloaded file
        """
        if dataset_name in self.DATASETS:
            dataset_info = self.DATASETS[dataset_name]
            filepath = self.data_dir / dataset_info['filename']
            
            # Skip if file exists and not forcing
            if filepath.exists() and not force:
                print(f"Dataset '{dataset_name}' already exists at {filepath}")
                return filepath
            
            print(f"Downloading {dataset_name} from {dataset_info['url']}")
            
            try:
                response = requests.get(dataset_info['url'], stream=True)
                response.raise_for_status()
                
                # Download with progress bar
                total_size = int(response.headers.get('content-length', 0))
                with open(filepath, 'wb') as f, tqdm(
                    desc=dataset_name,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for data in response.iter_content(chunk_size=8192):
                        f.write(data)
                        pbar.update(len(data))
                
                print(f"Successfully downloaded {dataset_name}")
                return filepath
                
            except Exception as e:
                print(f"Error downloading {dataset_name}: {str(e)}")
                return None
                
        elif dataset_name in self.ALTERNATIVE_SOURCES:
            print(f"\nDataset '{dataset_name}' requires manual download:")
            print(f"Source: {self.ALTERNATIVE_SOURCES[dataset_name]['source']}")
            print(f"Instructions: {self.ALTERNATIVE_SOURCES[dataset_name]['instructions']}")
            print("Please download and place in data/raw/ directory.")
            return None
            
        else:
            print(f"Unknown dataset: {dataset_name}")
            return None
    
    def download_all(self, force=False):
        """
        Download all available datasets.
        
        Parameters:
        -----------
        force : bool
            Force re-download of all datasets
            
        Returns:
        --------
        downloaded : dict
            Dictionary of downloaded file paths
        """
        downloaded = {}
        
        print("Downloading microarray datasets...")
        print("=" * 50)
        
        # Download automatically available datasets
        for dataset_name in self.DATASETS:
            print(f"\n{dataset_name.upper()}:")
            filepath = self.download_dataset(dataset_name, force)
            if filepath:
                downloaded[dataset_name] = filepath
        
        # Inform about manual downloads
        print("\n" + "=" * 50)
        print("Datasets requiring manual download:")
        print("=" * 50)
        
        for dataset_name, info in self.ALTERNATIVE_SOURCES.items():
            print(f"\n{dataset_name.upper()}:")
            print(f"  Source: {info['source']}")
            print(f"  Instructions: {info['instructions']}")
            
            # Check if already exists
            possible_files = [
                f"{dataset_name}.csv",
                f"{dataset_name}.txt",
                f"{dataset_name}.mat",
                f"{dataset_name}.xlsx"
            ]
            
            for filename in possible_files:
                if (self.data_dir / filename).exists():
                    downloaded[dataset_name] = self.data_dir / filename
                    print(f"  ✓ Found: {filename}")
                    break
        
        return downloaded
    
    def load_dataset(self, dataset_name):
        """
        Load a dataset into numpy arrays.
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to load
            
        Returns:
        --------
        X : numpy.ndarray
            Feature matrix (n_samples, n_features)
        y : numpy.ndarray
            Target labels
        """
        if dataset_name not in self.DATASETS and dataset_name not in self.ALTERNATIVE_SOURCES:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Check if dataset exists
        possible_extensions = ['.csv', '.mat', '.txt', '.xlsx']
        filepath = None
        
        for ext in possible_extensions:
            test_path = self.data_dir / f"{dataset_name}{ext}"
            if test_path.exists():
                filepath = test_path
                break
        
        if not filepath:
            # Try with original filename
            if dataset_name in self.DATASETS:
                filepath = self.data_dir / self.DATASETS[dataset_name]['filename']
            else:
                raise FileNotFoundError(f"Dataset '{dataset_name}' not found. Please download first.")
        
        # Load based on file extension
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
            # Assuming last column is target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
        elif filepath.suffix == '.mat':
            # For MATLAB files, use scipy
            from scipy.io import loadmat
            data = loadmat(filepath)
            
            # Try common key names
            possible_X_keys = ['X', 'data', 'fea', 'features']
            possible_y_keys = ['Y', 'labels', 'gnd', 'target']
            
            X = None
            y = None
            
            for key in possible_X_keys:
                if key in data:
                    X = data[key]
                    break
            
            for key in possible_y_keys:
                if key in data:
                    y = data[key].flatten()
                    break
            
            if X is None or y is None:
                raise ValueError(f"Could not find data/labels in {filepath}")
                
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Convert to appropriate types
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        
        print(f"Loaded {dataset_name}: X.shape={X.shape}, y.shape={y.shape}")
        return X, y
    
    def generate_synthetic_dataset(self, dataset_name, n_samples=None, n_features=None):
        """
        Generate synthetic data for testing when real data is not available.
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to simulate
        n_samples : int, optional
            Number of samples (uses paper values if None)
        n_features : int, optional
            Number of features (uses paper values if None)
            
        Returns:
        --------
        X : numpy.ndarray
            Synthetic feature matrix
        y : numpy.ndarray
            Synthetic labels
        """
        # Default sizes from paper
        dataset_sizes = {
            'leukemia': (72, 7129),
            'brain_cancer': (90, 10367),
            'colon_cancer': (62, 2000),
            'srbct': (83, 2308),
            'prostate_tumor': (102, 12600),
            'lung_cancer': (203, 12533),
            'lymphoma': (96, 4026),
            '11_tumors': (174, 4200),
            'dlbcl': (42, 3812)
        }
        
        if dataset_name not in dataset_sizes:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if n_samples is None:
            n_samples = dataset_sizes[dataset_name][0]
        if n_features is None:
            n_features = dataset_sizes[dataset_name][1]
        
        # Generate synthetic data with realistic properties
        np.random.seed(42)
        
        # Create informative features (10% of total)
        n_informative = max(1, int(n_features * 0.1))
        
        # Generate base features
        X = np.random.randn(n_samples, n_features)
        
        # Make some features informative
        for i in range(n_informative):
            if i % 3 == 0:
                # Linear relationship
                X[:, i] = X[:, i] * 2 + np.random.randn(n_samples) * 0.3
            elif i % 3 == 1:
                # Quadratic relationship
                X[:, i] = X[:, i] ** 2 + np.random.randn(n_samples) * 0.3
            else:
                # Log relationship
                X[:, i] = np.log(np.abs(X[:, i]) + 1) + np.random.randn(n_samples) * 0.3
        
        # Generate labels based on informative features
        y_base = np.sum(X[:, :n_informative], axis=1)
        y = (y_base > np.median(y_base)).astype(int)
        
        # Add some noise to labels (5% mislabeling)
        n_noisy = int(n_samples * 0.05)
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        y[noisy_indices] = 1 - y[noisy_indices]
        
        print(f"Generated synthetic {dataset_name}:")
        print(f"  Samples: {n_samples}, Features: {n_features}")
        print(f"  Informative features: {n_informative}")
        print(f"  Class distribution: {np.bincount(y)}")
        
        return X, y

def main():
    """Main function to download all datasets."""
    downloader = MicroarrayDatasetDownloader()
    
    print("Microarray Dataset Downloader")
    print("=" * 60)
    
    # List available datasets
    print("\nAvailable datasets:")
    print("-" * 30)
    
    for i, dataset in enumerate(downloader.DATASETS.keys(), 1):
        info = downloader.DATASETS[dataset]
        print(f"{i}. {dataset.upper()}: {info['description']}")
        print(f"   Features: {info['n_features']}, Samples: {info['n_samples']}")
    
    print("\nDatasets requiring manual download:")
    print("-" * 30)
    
    for i, dataset in enumerate(downloader.ALTERNATIVE_SOURCES.keys(), 1):
        info = downloader.ALTERNATIVE_SOURCES[dataset]
        print(f"{i}. {dataset.upper()}: {info['source']}")
    
    # Download all
    print("\n" + "=" * 60)
    response = input("Download all available datasets? (y/n): ")
    
    if response.lower() == 'y':
        downloaded = downloader.download_all()
        
        print("\nDownload summary:")
        print("-" * 30)
        
        for dataset, filepath in downloaded.items():
            if filepath:
                print(f"✓ {dataset.upper()}: {filepath}")
            else:
                print(f"✗ {dataset.upper()}: Not downloaded (manual required)")
    
    # Generate synthetic data option
    print("\n" + "=" * 60)
    response = input("Generate synthetic data for testing? (y/n): ")
    
    if response.lower() == 'y':
        synth_dir = Path('data/synthetic')
        synth_dir.mkdir(exist_ok=True)
        
        for dataset in downloader.DATASETS:
            X, y = downloader.generate_synthetic_dataset(dataset)
            
            # Save synthetic data
            synth_path = synth_dir / f"{dataset}_synthetic.npz"
            np.savez(synth_path, X=X, y=y)
            print(f"Saved synthetic {dataset} to {synth_path}")

if __name__ == "__main__":
    main()