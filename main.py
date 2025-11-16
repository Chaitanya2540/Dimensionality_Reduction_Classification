"""
Dimensionality Reduction for Classification Project
Main script to run the complete analysis pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_and_preprocess_data
from dimensionality_reduction import PCA_Reducer, LDA_Reducer
from classifiers import KNN_Classifier, Linear_Classifier, Mahalanobis_Classifier

def evaluate_classifier(classifier, X_train, y_train, X_test, y_test, name):
    """Train and evaluate a classifier, returning accuracy and timing"""
    # Training time
    start_time = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Prediction time
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    predict_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    total_time = train_time + predict_time
    
    return accuracy, y_pred, train_time, predict_time, total_time

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
    timing_results = {}
    info_retention = {}
    
    for method_name, reducer in reduction_methods.items():
        print(f"\n[Step 7] Running {method_name} experiments...")
        
        # Select appropriate dimensions
        if method_name == 'PCA':
            dims_to_test = pca_dims
        else:  # LDA
            dims_to_test = lda_dims
        
        results[method_name] = {}
        timing_results[method_name] = {}
        info_retention[method_name] = {}
        
        for dim in dims_to_test:
            print(f"  Testing dimension: {dim}")
            
            # Fit reducer on training data only
            X_train_reduced = reducer.fit_transform(X_train_scaled, y_train, dim)
            X_test_reduced = reducer.transform(X_test_scaled, dim)
            
            # Calculate information retention
            if method_name == 'PCA':
                # For PCA: cumulative explained variance ratio (sum of all components)
                explained_var = reducer.get_explained_variance_ratio()
                if explained_var is not None:
                    cumsum_var = np.sum(explained_var)  # Sum of explained variance for all components
                    info_retention[method_name][dim] = cumsum_var
                else:
                    info_retention[method_name][dim] = 0.0
            else:  # LDA
                # For LDA: we can't easily measure information retention
                # Use a placeholder or calculate class separation metric
                info_retention[method_name][dim] = dim / max_lda_dim  # Normalized dimension ratio
            
            results[method_name][dim] = {}
            timing_results[method_name][dim] = {}
            
            # Test each classifier
            for clf_name, classifier in classifiers.items():
                try:
                    accuracy, y_pred, train_time, predict_time, total_time = evaluate_classifier(
                        classifier, X_train_reduced, y_train, 
                        X_test_reduced, y_test, clf_name
                    )
                    results[method_name][dim][clf_name] = accuracy
                    timing_results[method_name][dim][clf_name] = {
                        'train_time': train_time,
                        'predict_time': predict_time,
                        'total_time': total_time
                    }
                    print(f"    {clf_name}: {accuracy:.4f} (Time: {total_time:.4f}s)")
                except Exception as e:
                    print(f"    {clf_name}: Error - {str(e)}")
                    results[method_name][dim][clf_name] = None
                    timing_results[method_name][dim][clf_name] = None
    
    # Step 8: Visualize results
    print("\n[Step 8] Generating visualizations...")
    plot_results(results, dataset_name, original_dim)
    plot_timing_results(timing_results, results, dataset_name)
    plot_info_retention(info_retention, results, dataset_name, original_dim)
    
    # Step 9: Save results summary
    print("\n[Step 9] Saving results summary...")
    save_results_summary(results, timing_results, info_retention, dataset_name, original_dim)
    
    # Step 10: Generate comparison table
    print("\n[Step 10] Generating comparison table...")
    generate_comparison_table(results, timing_results, info_retention, dataset_name, original_dim)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    
    return results, timing_results, info_retention

def plot_results(results, dataset_name, original_dim):
    """Create visualization plots for accuracy vs dimensionality"""
    
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
        if method_name == 'PCA':
            ax.axvline(x=original_dim, color='r', linestyle='--', alpha=0.5, label='Original Dim')
    
    plt.tight_layout()
    plt.savefig(f'results_{dataset_name}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved plot: results_{dataset_name}.png")
    plt.close()

def plot_timing_results(timing_results, results, dataset_name):
    """Create visualization plots for classification time vs dimensionality"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (method_name, method_timing) in enumerate(timing_results.items()):
        ax = axes[idx]
        
        dims = sorted(method_timing.keys())
        clf_names = list(timing_results[method_name][dims[0]].keys())
        
        for clf_name in clf_names:
            times = []
            valid_dims = []
            
            for dim in dims:
                timing = method_timing[dim].get(clf_name)
                if timing is not None:
                    times.append(timing['total_time'])
                    valid_dims.append(dim)
            
            if times:
                ax.plot(valid_dims, times, marker='o', label=clf_name, linewidth=2)
        
        ax.set_xlabel('Dimensionality', fontsize=12)
        ax.set_ylabel('Classification Time (seconds)', fontsize=12)
        ax.set_title(f'{method_name} - Classification Time vs Dimensionality', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_yscale('log')  # Log scale for better visualization
    
    # Efficiency plot (accuracy/time ratio)
    for idx, (method_name, method_timing) in enumerate(timing_results.items()):
        ax = axes[idx + 2]
        
        dims = sorted(method_timing.keys())
        clf_names = list(timing_results[method_name][dims[0]].keys())
        
        for clf_name in clf_names:
            efficiencies = []
            valid_dims = []
            
            for dim in dims:
                timing = method_timing[dim].get(clf_name)
                if timing is not None and timing['total_time'] > 0:
                    # Get corresponding accuracy
                    acc = results[method_name][dim].get(clf_name)
                    if acc is not None:
                        efficiency = acc / timing['total_time']  # Accuracy per second
                        efficiencies.append(efficiency)
                        valid_dims.append(dim)
            
            if efficiencies:
                ax.plot(valid_dims, efficiencies, marker='o', label=clf_name, linewidth=2)
        
        ax.set_xlabel('Dimensionality', fontsize=12)
        ax.set_ylabel('Efficiency (Accuracy / Time)', fontsize=12)
        ax.set_title(f'{method_name} - Classification Efficiency vs Dimensionality', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f'timing_efficiency_{dataset_name}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved plot: timing_efficiency_{dataset_name}.png")
    plt.close()

def plot_info_retention(info_retention, results, dataset_name, original_dim):
    """Create visualization plots for information retention vs accuracy"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (method_name, method_retention) in enumerate(info_retention.items()):
        ax = axes[idx]
        
        dims = sorted(method_retention.keys())
        clf_names = list(results[method_name][dims[0]].keys())
        
        for clf_name in clf_names:
            retentions = []
            accuracies = []
            
            for dim in dims:
                retention = method_retention.get(dim, 0)
                acc = results[method_name][dim].get(clf_name)
                if acc is not None:
                    retentions.append(retention)
                    accuracies.append(acc)
            
            if retentions:
                ax.scatter(retentions, accuracies, label=clf_name, s=100, alpha=0.6)
        
        ax.set_xlabel('Information Retention', fontsize=12)
        ax.set_ylabel('Classification Accuracy', fontsize=12)
        title = f'{method_name} - Accuracy vs Information Retention'
        if method_name == 'PCA':
            title += '\n(Explained Variance Ratio)'
        else:
            title += '\n(Normalized Dimension Ratio)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'info_retention_{dataset_name}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved plot: info_retention_{dataset_name}.png")
    plt.close()

def save_results_summary(results, timing_results, info_retention, dataset_name, original_dim):
    """Save results to a text file"""
    with open(f'results_summary_{dataset_name}.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("Dimensionality Reduction for Classification - Results Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Original Dimension: {original_dim}\n\n")
        
        for method_name, method_results in results.items():
            f.write(f"\n{method_name} Results:\n")
            f.write("="*80 + "\n")
            
            dims = sorted(method_results.keys())
            clf_names = list(results[method_name][dims[0]].keys())
            
            # Accuracy table
            f.write("\nAccuracy Results:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Dimension':<12}{'Reduction':<12}")
            for clf_name in clf_names:
                f.write(f"{clf_name:<18}")
            f.write("\n")
            f.write("-" * 80 + "\n")
            
            for dim in dims:
                reduction = original_dim - dim
                reduction_pct = (reduction / original_dim) * 100
                f.write(f"{dim:<12}{reduction_pct:>6.1f}%     ")
                for clf_name in clf_names:
                    acc = results[method_name][dim].get(clf_name)
                    if acc is not None:
                        f.write(f"{acc:<18.4f}")
                    else:
                        f.write(f"{'N/A':<18}")
                f.write("\n")
            
            # Timing table
            f.write("\n\nClassification Time (seconds):\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Dimension':<12}")
            for clf_name in clf_names:
                f.write(f"{clf_name:<18}")
            f.write("\n")
            f.write("-" * 80 + "\n")
            
            for dim in dims:
                f.write(f"{dim:<12}")
                for clf_name in clf_names:
                    timing = timing_results[method_name][dim].get(clf_name)
                    if timing is not None:
                        f.write(f"{timing['total_time']:<18.4f}")
                    else:
                        f.write(f"{'N/A':<18}")
                f.write("\n")
            
            # Information retention
            f.write("\n\nInformation Retention:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Dimension':<12}{'Retention':<12}\n")
            f.write("-" * 80 + "\n")
            for dim in dims:
                retention = info_retention[method_name].get(dim, 0)
                if method_name == 'PCA':
                    f.write(f"{dim:<12}{retention:<12.4f} (Explained Variance)\n")
                else:
                    f.write(f"{dim:<12}{retention:<12.4f} (Normalized Ratio)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"  Saved summary: results_summary_{dataset_name}.txt")

def generate_comparison_table(results, timing_results, info_retention, dataset_name, original_dim):
    """Generate comprehensive comparison table between PCA and LDA"""
    
    with open(f'comparison_table_{dataset_name}.txt', 'w') as f:
        f.write("="*100 + "\n")
        f.write("COMPREHENSIVE COMPARISON: PCA vs LDA with Different Classifiers\n")
        f.write("="*100 + "\n\n")
        f.write(f"Original Dimension: {original_dim}\n\n")
        
        # Find best configurations
        clf_names = ['k-NN (k=3)', 'k-NN (k=5)', 'Linear SVM', 'Mahalanobis']
        
        f.write("Best Configuration for Each Method-Classifier Combination:\n")
        f.write("="*100 + "\n")
        f.write(f"{'Method':<8}{'Classifier':<20}{'Best Dim':<12}{'Reduction':<15}{'Accuracy':<12}")
        f.write(f"{'Time (s)':<12}{'Efficiency':<15}{'Info Retention':<15}\n")
        f.write("-"*100 + "\n")
        
        for method_name in ['PCA', 'LDA']:
            for clf_name in clf_names:
                best_acc = 0
                best_dim = 0
                best_timing = None
                best_retention = 0
                
                dims = sorted(results[method_name].keys())
                for dim in dims:
                    acc = results[method_name][dim].get(clf_name)
                    if acc is not None and acc > best_acc:
                        best_acc = acc
                        best_dim = dim
                        best_timing = timing_results[method_name][dim].get(clf_name)
                        best_retention = info_retention[method_name].get(dim, 0)
                
                if best_dim > 0:
                    reduction = original_dim - best_dim
                    reduction_pct = (reduction / original_dim) * 100
                    time_val = best_timing['total_time'] if best_timing else 0
                    efficiency = best_acc / time_val if time_val > 0 else 0
                    
                    f.write(f"{method_name:<8}{clf_name:<20}{best_dim:<12}")
                    f.write(f"{reduction_pct:>6.1f}% ({reduction}){'':<6}")
                    f.write(f"{best_acc:<12.4f}")
                    f.write(f"{time_val:<12.4f}")
                    f.write(f"{efficiency:<15.2f}")
                    f.write(f"{best_retention:<15.4f}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("\nSummary Statistics:\n")
        f.write("="*100 + "\n")
        
        # Overall best for each method
        for method_name in ['PCA', 'LDA']:
            f.write(f"\n{method_name} - Overall Best Performance:\n")
            f.write("-"*100 + "\n")
            
            best_overall_acc = 0
            best_overall_config = None
            
            for clf_name in clf_names:
                dims = sorted(results[method_name].keys())
                for dim in dims:
                    acc = results[method_name][dim].get(clf_name)
                    if acc is not None and acc > best_overall_acc:
                        best_overall_acc = acc
                        timing = timing_results[method_name][dim].get(clf_name)
                        retention = info_retention[method_name].get(dim, 0)
                        best_overall_config = {
                            'clf': clf_name,
                            'dim': dim,
                            'acc': acc,
                            'timing': timing,
                            'retention': retention
                        }
            
            if best_overall_config:
                cfg = best_overall_config
                reduction = original_dim - cfg['dim']
                reduction_pct = (reduction / original_dim) * 100
                time_val = cfg['timing']['total_time'] if cfg['timing'] else 0
                efficiency = cfg['acc'] / time_val if time_val > 0 else 0
                
                f.write(f"  Classifier: {cfg['clf']}\n")
                f.write(f"  Dimension: {cfg['dim']} (Reduction: {reduction_pct:.1f}% or {reduction} dimensions)\n")
                f.write(f"  Accuracy: {cfg['acc']:.4f}\n")
                f.write(f"  Classification Time: {time_val:.4f} seconds\n")
                f.write(f"  Efficiency: {efficiency:.2f} (accuracy/time)\n")
                f.write(f"  Information Retention: {cfg['retention']:.4f}\n")
        
        f.write("\n" + "="*100 + "\n")
    
    print(f"  Saved comparison table: comparison_table_{dataset_name}.txt")

if __name__ == "__main__":
    # Run the analysis
    # You can change dataset_name to 'synthetic' if you want to use synthetic data
    results, timing_results, info_retention = run_analysis(dataset_name='mnist', n_samples=5000, test_size=0.2, random_state=42)

