"""
Dimensionality reduction module
Implements PCA and LDA
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class PCA_Reducer:
    """
    Principal Component Analysis (PCA) for dimensionality reduction
    """
    
    def __init__(self):
        self.pca = None
        self.fitted_dim = None
    
    def fit_transform(self, X, y=None, n_components=None):
        """
        Fit PCA and transform data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target labels (not used in PCA, but kept for interface consistency)
        n_components : int
            Number of components to keep
        
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        if n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        
        self.pca = PCA(n_components=n_components, random_state=42)
        X_transformed = self.pca.fit_transform(X)
        self.fitted_dim = n_components
        
        return X_transformed
    
    def transform(self, X, n_components=None):
        """
        Transform data using fitted PCA
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
        n_components : int, optional
            Number of components (uses fitted value if not provided)
        
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted before transform")
        
        if n_components is None:
            n_components = self.fitted_dim
        
        # Use only the first n_components
        X_transformed = self.pca.transform(X)
        if X_transformed.shape[1] > n_components:
            X_transformed = X_transformed[:, :n_components]
        
        return X_transformed
    
    def get_explained_variance_ratio(self):
        """Get explained variance ratio for each component"""
        if self.pca is None:
            return None
        return self.pca.explained_variance_ratio_

class LDA_Reducer:
    """
    Linear Discriminant Analysis (LDA) for dimensionality reduction
    """
    
    def __init__(self):
        self.lda = None
        self.fitted_dim = None
    
    def fit_transform(self, X, y, n_components=None):
        """
        Fit LDA and transform data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels (required for LDA)
        n_components : int
            Number of components to keep (max is n_classes - 1)
        
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        n_classes = len(np.unique(y))
        max_components = min(n_classes - 1, X.shape[1], X.shape[0] - 1)
        
        if n_components is None:
            n_components = max_components
        else:
            n_components = min(n_components, max_components)
        
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_transformed = self.lda.fit_transform(X, y)
        self.fitted_dim = n_components
        
        return X_transformed
    
    def transform(self, X, n_components=None):
        """
        Transform data using fitted LDA
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
        n_components : int, optional
            Number of components (uses fitted value if not provided)
        
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        if self.lda is None:
            raise ValueError("LDA must be fitted before transform")
        
        if n_components is None:
            n_components = self.fitted_dim
        
        X_transformed = self.lda.transform(X)
        if X_transformed.shape[1] > n_components:
            X_transformed = X_transformed[:, :n_components]
        
        return X_transformed

