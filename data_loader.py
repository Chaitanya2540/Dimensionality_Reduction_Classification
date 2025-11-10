"""
Data loading and preprocessing module
"""

import numpy as np
from sklearn.datasets import fetch_openml, make_classification
from sklearn.preprocessing import LabelEncoder

def load_mnist(n_samples=5000, random_state=42):
    """
    Load MNIST dataset from OpenML
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to load
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target labels
    """
    print("  Loading MNIST dataset from OpenML...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target
    
    # Convert labels to integers
    y = y.astype(int)
    
    # Sample data if needed
    if n_samples < len(X):
        np.random.seed(random_state)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    print(f"  Loaded {len(X)} samples with {X.shape[1]} features")
    return X, y

def generate_synthetic_data(n_samples=5000, n_features=784, n_classes=10, random_state=42):
    """
    Generate synthetic high-dimensional classification data
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features (dimensions)
    n_classes : int
        Number of classes
    random_state : int
        Random seed
    
    Returns:
    --------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target labels
    """
    print("  Generating synthetic high-dimensional data...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.1),  # 10% informative features
        n_redundant=int(n_features * 0.05),    # 5% redundant features
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
        class_sep=2.0  # Separation between classes
    )
    
    print(f"  Generated {len(X)} samples with {X.shape[1]} features")
    return X, y

def load_and_preprocess_data(dataset_name='mnist', n_samples=5000, random_state=42):
    """
    Load and preprocess data based on dataset name
    
    Parameters:
    -----------
    dataset_name : str
        'mnist' or 'synthetic'
    n_samples : int
        Number of samples to use
    random_state : int
        Random seed
    
    Returns:
    --------
    X : array-like, shape (n_samples, n_features)
        Preprocessed feature matrix
    y : array-like, shape (n_samples,)
        Target labels
    """
    if dataset_name.lower() == 'mnist':
        X, y = load_mnist(n_samples, random_state)
    elif dataset_name.lower() == 'synthetic':
        X, y = generate_synthetic_data(n_samples, n_features=784, n_classes=10, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'mnist' or 'synthetic'")
    
    # Additional preprocessing if needed
    # Remove any NaN or infinite values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure labels are integers starting from 0
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y

