"""
Dimensionality Reduction for Classification Project
Main script to run the complete analysis pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_and_preprocess_data
from dimensionality_reduction import PCA_Reducer, LDA_Reducer
from classifiers import KNN_Classifier, Linear_Classifier, Mahalanobis_Classifier

def evaluate_classifier(classifier, X_train, y_train, X_test, y_test, name):
    """Train and evaluate a classifier"""
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

def run_analysis(dataset_name='mnist', n_samples=5000, test_size=0.2, random_state=42):
    """
    Main analysis pipeline
    
    Parameters:
    -----------
    dataset_name : str
        Name of dataset ('mnist' or 'synthetic')
    n_samples : int
        Number of samples to use (for faster computation)
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    """
    print("="*60)
    print("Dimensionality Reduction for Classification Analysis")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    X, y = load_and_preprocess_data(dataset_name, n_samples, random_state)
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Step 2: Split into training and testing sets
    print("\n[Step 2] Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Step 3: Normalize data (using training statistics only)
    print("\n[Step 3] Normalizing data (using training statistics)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 4: Define dimensionality reduction methods
    reduction_methods = {
        'PCA': PCA_Reducer(),
        'LDA': LDA_Reducer()
    }
    
    # Step 5: Define classifiers
    classifiers = {
        'k-NN (k=3)': KNN_Classifier(n_neighbors=3),
        'k-NN (k=5)': KNN_Classifier(n_neighbors=5),
        'Linear SVM': Linear_Classifier(),
        'Mahalanobis': Mahalanobis_Classifier()
    }
    
    # Step 6: Define dimensions to test
    original_dim = X_train_scaled.shape[1]
    n_classes = len(np.unique(y_train))
    
    # For PCA: test various dimensions
    pca_dims = [2, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    pca_dims = [d for d in pca_dims if d < original_dim]
    pca_dims.append(original_dim)  # Include original dimension
    
    # For LDA: maximum is (n_classes - 1)
    max_lda_dim = min(n_classes - 1, original_dim)
    lda_dims = [2, 3, 4, 5, 6, 7, 8, 9]
    lda_dims = [d for d in lda_dims if d <= max_lda_dim]
    
    # Step 7: Run experiments
    results = {}
    
    for method_name, reducer in reduction_methods.items():
        print(f"\n[Step 7] Running {method_name} experiments...")
        
        # Select appropriate dimensions
        if method_name == 'PCA':
            dims_to_test = pca_dims
        else:  # LDA
            dims_to_test = lda_dims
        
        results[method_name] = {}
        
        for dim in dims_to_test:
            print(f"  Testing dimension: {dim}")
            
            # Fit reducer on training data only
            X_train_reduced = reducer.fit_transform(X_train_scaled, y_train, dim)
            X_test_reduced = reducer.transform(X_test_scaled, dim)
            
            results[method_name][dim] = {}
            
            # Test each classifier
            for clf_name, classifier in classifiers.items():
                try:
                    accuracy, y_pred = evaluate_classifier(
                        classifier, X_train_reduced, y_train, 
                        X_test_reduced, y_test, clf_name
                    )
                    results[method_name][dim][clf_name] = accuracy
                    print(f"    {clf_name}: {accuracy:.4f}")
                except Exception as e:
                    print(f"    {clf_name}: Error - {str(e)}")
                    results[method_name][dim][clf_name] = None
    
    # Step 8: Visualize results
    print("\n[Step 8] Generating visualizations...")
    plot_results(results, dataset_name)
    
    # Step 9: Save results summary
    print("\n[Step 9] Saving results summary...")
    save_results_summary(results, dataset_name)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    
    return results

def plot_results(results, dataset_name):
    """Create visualization plots"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (method_name, method_results) in enumerate(results.items()):
        ax = axes[idx]
        
        dims = sorted(method_results.keys())
        
        for clf_name in results[method_name][dims[0]].keys():
            accuracies = []
            valid_dims = []
            
            for dim in dims:
                acc = results[method_name][dim].get(clf_name)
                if acc is not None:
                    accuracies.append(acc)
                    valid_dims.append(dim)
            
            if accuracies:
                ax.plot(valid_dims, accuracies, marker='o', label=clf_name, linewidth=2)
        
        ax.set_xlabel('Dimensionality', fontsize=12)
        ax.set_ylabel('Classification Accuracy', fontsize=12)
        ax.set_title(f'{method_name} - Accuracy vs Dimensionality', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'results_{dataset_name}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved plot: results_{dataset_name}.png")
    plt.close()

def save_results_summary(results, dataset_name):
    """Save results to a text file"""
    with open(f'results_summary_{dataset_name}.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("Dimensionality Reduction for Classification - Results Summary\n")
        f.write("="*60 + "\n\n")
        
        for method_name, method_results in results.items():
            f.write(f"\n{method_name} Results:\n")
            f.write("-" * 60 + "\n")
            
            dims = sorted(method_results.keys())
            
            # Header
            f.write(f"{'Dimension':<12}")
            for clf_name in results[method_name][dims[0]].keys():
                f.write(f"{clf_name:<20}")
            f.write("\n")
            f.write("-" * 60 + "\n")
            
            # Data rows
            for dim in dims:
                f.write(f"{dim:<12}")
                for clf_name in results[method_name][dims[0]].keys():
                    acc = results[method_name][dim].get(clf_name)
                    if acc is not None:
                        f.write(f"{acc:<20.4f}")
                    else:
                        f.write(f"{'N/A':<20}")
                f.write("\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"  Saved summary: results_summary_{dataset_name}.txt")

if __name__ == "__main__":
    # Run the analysis
    # You can change dataset_name to 'synthetic' if you want to use synthetic data
    results = run_analysis(dataset_name='mnist', n_samples=5000, test_size=0.2, random_state=42)

