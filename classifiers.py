"""
Classification module
Implements k-NN, Linear SVM, and Mahalanobis Distance classifiers
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv, pinv

class KNN_Classifier:
    """
    k-Nearest Neighbors Classifier
    """
    
    def __init__(self, n_neighbors=5):
        """
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors to use
        """
        self.n_neighbors = n_neighbors
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    def fit(self, X, y):
        """Train the classifier"""
        self.clf.fit(X, y)
    
    def predict(self, X):
        """Make predictions"""
        return self.clf.predict(X)

class Linear_Classifier:
    """
    Linear Support Vector Machine Classifier
    """
    
    def __init__(self, C=1.0):
        """
        Parameters:
        -----------
        C : float
            Regularization parameter
        """
        self.C = C
        self.clf = SVC(kernel='linear', C=C, random_state=42)
    
    def fit(self, X, y):
        """Train the classifier"""
        self.clf.fit(X, y)
    
    def predict(self, X):
        """Make predictions"""
        return self.clf.predict(X)

class Mahalanobis_Classifier:
    """
    Minimum Mahalanobis Distance Classifier
    """
    
    def __init__(self):
        self.class_means = {}
        self.covariances = {}
        self.classes = None
    
    def fit(self, X, y):
        """
        Train the classifier by computing class means and covariances
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Training labels
        """
        self.classes = np.unique(y)
        self.class_means = {}
        self.covariances = {}
        
        for cls in self.classes:
            # Get samples for this class
            X_cls = X[y == cls]
            
            # Compute mean
            self.class_means[cls] = np.mean(X_cls, axis=0)
            
            # Compute covariance matrix
            # Add small regularization to avoid singular matrix
            cov = np.cov(X_cls.T)
            # Regularize with small value to ensure invertibility
            reg = 1e-6 * np.eye(cov.shape[0])
            self.covariances[cls] = cov + reg
    
    def predict(self, X):
        """
        Predict using minimum Mahalanobis distance
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        
        Returns:
        --------
        predictions : array-like, shape (n_samples,)
            Predicted labels
        """
        if self.classes is None:
            raise ValueError("Classifier must be fitted before prediction")
        
        predictions = []
        
        for x in X:
            min_distance = np.inf
            predicted_class = self.classes[0]
            
            for cls in self.classes:
                mean = self.class_means[cls]
                cov = self.covariances[cls]
                
                try:
                    # Compute Mahalanobis distance
                    # Use pseudo-inverse if matrix is singular
                    try:
                        cov_inv = inv(cov)
                    except:
                        cov_inv = pinv(cov)
                    
                    diff = x - mean
                    distance = np.sqrt(diff.T @ cov_inv @ diff)
                    
                    if distance < min_distance:
                        min_distance = distance
                        predicted_class = cls
                
                except Exception as e:
                    # Fallback to Euclidean distance if Mahalanobis fails
                    distance = np.linalg.norm(x - mean)
                    if distance < min_distance:
                        min_distance = distance
                        predicted_class = cls
            
            predictions.append(predicted_class)
        
        return np.array(predictions)

