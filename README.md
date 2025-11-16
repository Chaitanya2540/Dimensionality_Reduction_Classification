# Dimensionality Reduction for Classification Project

This project explores how dimensionality reduction techniques (PCA and LDA) affect classification performance using various classifiers.

## Project Structure

```
Mini_Project/
├── main.py                      # Main analysis script
├── data_loader.py               # Data loading and preprocessing
├── dimensionality_reduction.py  # PCA and LDA implementations
├── classifiers.py              # k-NN, Linear SVM, and Mahalanobis classifiers
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Features

### Dimensionality Reduction Methods
- **PCA (Principal Component Analysis)**: Unsupervised dimensionality reduction
- **LDA (Linear Discriminant Analysis)**: Supervised dimensionality reduction

### Classifiers
- **k-Nearest Neighbors (k-NN)**: With k=3 and k=5
- **Linear SVM**: Linear support vector machine
- **Mahalanobis Distance**: Minimum Mahalanobis distance classifier

### Dataset
- **MNIST**: Handwritten digit recognition dataset (784 features)
- **Synthetic Data**: Option to generate synthetic high-dimensional data

## Installation

**Recommended: Use a virtual environment** to avoid conflicts with system packages.

### Option 1: Using venv (Python 3.3+)

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
   - **On macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
   - **On Windows:**
     ```bash
     venv\Scripts\activate
     ```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. When you're done, deactivate the virtual environment:
```bash
deactivate
```

### Option 2: Using conda

1. Create a conda environment:
```bash
conda create -n dimred_classification python=3.9
conda activate dimred_classification
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Direct Installation (Not Recommended)

If you prefer not to use a virtual environment (not recommended):
```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis script:

```bash
python main.py
```

The script will:
1. Load and preprocess the data
2. Split into training and testing sets
3. Normalize the data (using training statistics only)
4. Apply PCA and LDA for various dimensions
5. Evaluate each classifier at each dimensionality
6. Generate visualization plots
7. Save results summary

## Output

The script generates:
- `results_mnist.png`: Visualization plots showing accuracy vs dimensionality
- `results_summary_mnist.txt`: Text summary of all results

## Customization

Main script can be modified to:
- Change the dataset (set `dataset_name='synthetic'` for synthetic data)
- Adjust the number of samples (`n_samples` parameter)
- Modify the dimensions to test
- Add more classifiers or dimensionality reduction methods

## Key Implementation Details

1. **Proper Train/Test Split**: All parameters are learned only from training data
2. **Normalization**: Data is normalized using training statistics, then applied to test data
3. **Dimensionality Reduction**: Fitted on training data, then applied to both train and test
4. **Evaluation**: Only test data is used for final accuracy computation

## Results Analysis

The generated plots show:
- How classification accuracy changes with dimensionality
- Comparison between PCA and LDA
- Performance of different classifiers at various dimensions

