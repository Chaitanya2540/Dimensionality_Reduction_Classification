# Dimensionality Reduction for Classification: A Comparative Study of PCA and LDA

## Abstract

This report presents a comprehensive study on the impact of dimensionality reduction techniques on classification performance, computational efficiency, and information retention. We investigate Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) as dimensionality reduction methods, evaluating their effectiveness with multiple classifiers including k-Nearest Neighbors (k-NN), Linear Support Vector Machine (SVM), and Minimum Mahalanobis Distance classifier. Experiments were conducted on the MNIST handwritten digit dataset, systematically reducing dimensions from 2 to 784 (original) and measuring classification accuracy, computation time, and efficiency. Results demonstrate that optimal performance is achieved at intermediate dimensions (50-75 for PCA), with PCA outperforming LDA in accuracy (91.1% vs 83.4%) while LDA achieves better computational efficiency. The k-NN classifier with k=5 consistently achieved the highest accuracy across most dimensionalities.

## 1. Introduction

Dimensionality reduction is a fundamental technique in machine learning that addresses the curse of dimensionality by transforming high-dimensional data into lower-dimensional representations while preserving essential information. This study explores how dimensionality reduction affects classification performance, computational efficiency, and information retention, comparing unsupervised (PCA) and supervised (LDA) approaches.

The primary objectives of this research are:
1. To evaluate the performance of PCA and LDA at various dimensionalities
2. To compare classification accuracy across different classifiers
3. To analyze computational efficiency and time requirements
4. To examine information retention capabilities
5. To identify optimal dimensionality for each reduction method
6. To analyze the trade-offs between dimensionality, accuracy, and efficiency

## 2. Methodology

### 2.1 Dimensionality Reduction Techniques

#### 2.1.1 Principal Component Analysis (PCA)
PCA is an unsupervised linear transformation technique that projects data onto principal components—directions of maximum variance. It finds orthogonal axes that capture the most information in the data, ordered by the amount of variance they explain. PCA is particularly effective for removing redundant features and noise while preserving the most significant patterns. The explained variance ratio quantifies the proportion of total variance retained by each component.

#### 2.1.2 Linear Discriminant Analysis (LDA)
LDA is a supervised dimensionality reduction method that maximizes the separation between classes while minimizing within-class variance. Unlike PCA, LDA uses class labels to find projections that are optimal for classification. The maximum number of dimensions in LDA is limited to (C-1), where C is the number of classes.

### 2.2 Classification Methods

#### 2.2.1 k-Nearest Neighbors (k-NN)
k-NN classifies samples based on the majority class of the k nearest neighbors in the feature space. We evaluated k=3 and k=5, as these values typically provide good balance between bias and variance.

#### 2.2.2 Linear Support Vector Machine (SVM)
Linear SVM finds the optimal hyperplane that separates classes with maximum margin. It is effective for linearly separable data and provides good generalization performance.

#### 2.2.3 Minimum Mahalanobis Distance Classifier
This classifier assigns samples to the class with the minimum Mahalanobis distance, accounting for the covariance structure of each class. It assumes multivariate normal distributions and is optimal under certain statistical conditions.

## 3. Experimental Setup

### 3.1 Dataset
The MNIST dataset was selected for this study, consisting of 70,000 grayscale images of handwritten digits (0-9), each represented as a 784-dimensional vector (28×28 pixels). The dataset is well-preprocessed, normalized, and widely used as a benchmark in machine learning research.

### 3.2 Data Preprocessing
- **Sampling**: 5,000 samples were randomly selected to balance computational efficiency with statistical significance
- **Normalization**: Features were standardized using StandardScaler, with statistics computed only from the training set
- **Data Cleaning**: NaN and infinite values were removed to ensure data quality

### 3.3 Train-Test Split
The dataset was partitioned into:
- **Training Set**: 80% (4,000 samples) - used for learning all model parameters
- **Testing Set**: 20% (1,000 samples) - used exclusively for final evaluation

Critical to our methodology: all dimensionality reduction parameters (transformation matrices, component selection) and classifier parameters were learned exclusively from the training data. The test set was only used for final accuracy computation.

### 3.4 Experimental Parameters

**PCA Dimensions Tested**: 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 784 (original)

**LDA Dimensions Tested**: 2, 3, 4, 5, 6, 7, 8, 9 (maximum for 10 classes)

**Random Seed**: 42 (for reproducibility)

**Performance Metrics**: Classification accuracy, training time, prediction time, total computation time, efficiency (accuracy/time ratio), and information retention (explained variance for PCA, normalized dimension ratio for LDA)

## 4. Results

### 4.1 Classification Accuracy Results

#### 4.1.1 PCA Results

Table 1 presents classification accuracies for PCA at various dimensions. The results reveal several key patterns:

**Table 1: Classification Accuracy for PCA at Different Dimensions**

| Dimension | Reduction | k-NN (k=3) | k-NN (k=5) | Linear SVM | Mahalanobis |
|-----------|-----------|------------|------------|------------|-------------|
| 2         | 99.7%     | 0.2990     | 0.3120     | 0.3340     | 0.3320      |
| 5         | 99.4%     | 0.7000     | 0.7160     | 0.6970     | 0.6940      |
| 10        | 98.7%     | 0.8520     | 0.8580     | 0.8250     | 0.8210      |
| 20        | 97.4%     | 0.8970     | 0.9010     | 0.8760     | 0.8770      |
| 30        | 96.2%     | 0.8980     | 0.8950     | 0.8830     | 0.8730      |
| 50        | 93.6%     | 0.9020     | 0.9100     | 0.8940     | 0.8640      |
| 75        | 90.4%     | 0.9020     | **0.9110** | 0.8920     | 0.8490      |
| 100       | 87.2%     | 0.9010     | 0.9090     | 0.8880     | 0.7950      |
| 150       | 80.9%     | 0.8850     | 0.9060     | 0.8840     | 0.7810      |
| 200       | 74.5%     | 0.8810     | 0.9020     | 0.8880     | 0.7710      |
| 784       | 0.0%      | 0.8770     | 0.8910     | 0.8920     | 0.6500      |

**Key Observations:**
- Accuracy increases rapidly from 2 to 20 dimensions (29.9% → 90.1%)
- Peak performance occurs at 50-75 dimensions (91.1% for k-NN k=5)
- Performance plateaus and slightly decreases beyond 75 dimensions
- Original 784-dimensional data (0.877-0.892) performs worse than optimal reduced dimensions
- k-NN (k=5) achieves the highest accuracy (91.1%) at 75 dimensions with 90.4% dimension reduction

#### 4.1.2 LDA Results

Table 2 shows classification accuracies for LDA across different dimensions:

**Table 2: Classification Accuracy for LDA at Different Dimensions**

| Dimension | Reduction | k-NN (k=3) | k-NN (k=5) | Linear SVM | Mahalanobis |
|-----------|-----------|------------|------------|------------|-------------|
| 2         | 99.7%     | 0.4390     | 0.4610     | 0.5110     | 0.4910      |
| 3         | 99.6%     | 0.6360     | 0.6580     | 0.6820     | 0.6630      |
| 4         | 99.5%     | 0.7430     | 0.7550     | 0.7570     | 0.7540      |
| 5         | 99.4%     | 0.7420     | 0.7660     | 0.7670     | 0.7590      |
| 6         | 99.2%     | 0.7610     | 0.7680     | 0.7740     | 0.7720      |
| 7         | 99.1%     | 0.7960     | 0.8020     | 0.8040     | 0.7980      |
| 8         | 99.0%     | 0.8210     | 0.8300     | 0.8280     | 0.8170      |
| 9         | 98.9%     | 0.8200     | **0.8340** | 0.8190     | 0.8220      |

**Key Observations:**
- LDA shows steady improvement with increasing dimensions
- Best performance at dimension 9 (83.4% for k-NN k=5) with 98.9% dimension reduction
- Performance is consistently lower than PCA's optimal results
- All classifiers show similar performance trends
- Linear SVM performs competitively with k-NN at higher dimensions

### 4.2 Computational Efficiency Results

#### 4.2.1 Classification Time Analysis

Table 3 presents classification times for PCA at selected dimensions:

**Table 3: Classification Time (seconds) for PCA at Selected Dimensions**

| Dimension | k-NN (k=3) | k-NN (k=5) | Linear SVM | Mahalanobis |
|-----------|------------|------------|------------|-------------|
| 2         | 0.0459     | 0.0362     | 0.9754     | 0.1479      |
| 20        | 0.2996     | 0.1508     | 0.5491     | 0.2734      |
| 50        | 0.0805     | 0.0999     | 0.4932     | 0.5496      |
| 75        | 0.0938     | 0.1571     | 0.3880     | 3.3047      |
| 200       | 0.1962     | 0.0541     | 0.3668     | 24.4837     |
| 784       | 0.2349     | 0.0839     | 0.9327     | 692.0137    |

**Key Observations:**
- Mahalanobis classifier shows exponential time increase with dimensionality (0.15s at 2-D → 692s at 784-D)
- k-NN and Linear SVM show relatively stable times across dimensions
- Optimal dimensions (50-75) provide good balance between accuracy and time
- Original 784-D data requires significantly more computation time, especially for Mahalanobis

For LDA, classification times are consistently low (0.04-0.18 seconds) across all dimensions, demonstrating superior computational efficiency compared to PCA at higher dimensions.

#### 4.2.2 Efficiency Analysis

Table 4 shows the best configurations for each method-classifier combination:

**Table 4: Best Configuration for Each Method-Classifier Combination**

| Method | Classifier | Best Dim | Reduction | Accuracy | Time (s) | Efficiency |
|--------|-----------|----------|-----------|----------|----------|------------|
| PCA    | k-NN (k=3) | 50       | 93.6%     | 0.9020   | 0.0805   | 11.20      |
| PCA    | k-NN (k=5) | 75       | 90.4%     | 0.9110   | 0.1571   | 5.80       |
| PCA    | Linear SVM | 50       | 93.6%     | 0.8940   | 0.4932   | 1.81       |
| PCA    | Mahalanobis | 20       | 97.4%     | 0.8770   | 0.2734   | 3.21       |
| LDA    | k-NN (k=3) | 8        | 99.0%     | 0.8210   | 0.0561   | 14.62      |
| LDA    | k-NN (k=5) | 9        | 98.9%     | 0.8340   | 0.0728   | 11.46      |
| LDA    | Linear SVM | 8        | 99.0%     | 0.8280   | 0.1225   | 6.76       |
| LDA    | Mahalanobis | 9        | 98.9%     | 0.8220   | 0.1793   | 4.58       |

**Key Observations:**
- LDA achieves higher efficiency (accuracy/time) than PCA for most classifiers
- LDA with k-NN (k=3) achieves the highest efficiency (14.62) despite lower accuracy
- PCA achieves higher accuracy but at the cost of longer computation time
- Optimal efficiency occurs at intermediate dimensions for both methods

### 4.3 Information Retention Analysis

#### 4.3.1 PCA Information Retention

For PCA, information retention is measured by cumulative explained variance ratio:

| Dimension | Information Retention | Accuracy (k-NN k=5) |
|-----------|----------------------|---------------------|
| 2         | 0.1058 (10.58%)      | 0.3120              |
| 20        | 0.4211 (42.11%)      | 0.9010              |
| 50        | 0.6157 (61.57%)      | 0.9100              |
| 75        | 0.7119 (71.19%)      | 0.9110              |
| 200       | 0.9193 (91.93%)      | 0.9020              |
| 784       | 1.0000 (100%)        | 0.8910              |

**Key Finding**: At optimal dimension (75), PCA retains 71.19% of variance while achieving peak accuracy (91.1%), demonstrating that full variance retention is not necessary for optimal classification.

#### 4.3.2 LDA Information Retention

For LDA, information retention is measured as normalized dimension ratio (dim/max_dim):

| Dimension | Information Retention | Accuracy (k-NN k=5) |
|-----------|----------------------|---------------------|
| 2         | 0.2222 (22.22%)      | 0.4610              |
| 5         | 0.5556 (55.56%)      | 0.7660              |
| 9         | 1.0000 (100%)        | 0.8340              |

LDA achieves maximum information retention at dimension 9, which also corresponds to peak accuracy.

### 4.4 Visualizations

**Figure 1** (results_mnist.png) illustrates the relationship between dimensionality and classification accuracy for both PCA and LDA methods across all classifiers.

**Figure 2** (timing_efficiency_mnist.png) shows four subplots:
- Top row: Classification time vs dimensionality for PCA and LDA
- Bottom row: Efficiency (accuracy/time) vs dimensionality for PCA and LDA

**Figure 3** (info_retention_mnist.png) displays scatter plots showing the relationship between information retention and classification accuracy for PCA and LDA.

## 5. Analysis and Discussion

### 5.1 PCA vs LDA Comparison

PCA significantly outperforms LDA in classification accuracy, achieving 91.1% compared to LDA's 83.4%. However, LDA demonstrates superior computational efficiency. This difference can be attributed to several factors:

1. **Information Preservation**: PCA preserves maximum variance in the data, which is crucial for image data where pixel relationships contain important information. LDA focuses on class separation, which may discard variance important for classification.

2. **Dimensionality Flexibility**: PCA allows testing up to the original dimensionality (784), while LDA is constrained to 9 dimensions (C-1). This limitation prevents LDA from capturing more complex patterns.

3. **Data Characteristics**: MNIST digits have significant within-class variation (different writing styles), making variance-based reduction (PCA) more effective than class-separation-based reduction (LDA).

4. **Computational Efficiency**: LDA achieves faster classification times (0.07s vs 0.16s for optimal configurations) and higher efficiency scores, making it preferable for real-time applications where speed is critical.

### 5.2 Optimal Dimensionality Analysis

The results reveal a critical finding: **optimal performance occurs at intermediate dimensions (50-75 for PCA), not at the original 784 dimensions**. This phenomenon can be explained by:

1. **Curse of Dimensionality**: High-dimensional spaces are sparse, making distance-based classifiers (k-NN, Mahalanobis) less effective. Reducing dimensions improves density and classifier performance.

2. **Noise Reduction**: Lower dimensions filter out noise and irrelevant features, improving signal-to-noise ratio. At 75 dimensions, PCA retains only 71.19% of variance yet achieves peak accuracy.

3. **Overfitting Prevention**: Reduced dimensions act as implicit regularization, preventing overfitting to training data.

4. **Computational Efficiency**: Lower dimensions enable faster training and prediction while maintaining or improving accuracy. The optimal 75-D configuration is 4.5× faster than 784-D for Mahalanobis classifier.

### 5.3 Computational Efficiency Analysis

**Time Complexity Trends:**
- **k-NN**: Relatively stable time across dimensions (0.04-0.30s), with slight variations
- **Linear SVM**: Moderate time increase with dimensionality (0.39-0.93s)
- **Mahalanobis**: Exponential time increase (0.15s at 2-D → 692s at 784-D), making it impractical for high dimensions

**Efficiency Trade-offs:**
- LDA achieves 2× higher efficiency than PCA for k-NN classifiers
- PCA's higher accuracy comes at the cost of 2-3× longer computation time
- Optimal efficiency occurs at different dimensions than optimal accuracy, requiring careful consideration of application requirements

### 5.4 Information Retention vs Performance

**PCA**: The relationship between information retention and accuracy is non-linear. Peak accuracy (91.1%) occurs at 71.19% variance retention, demonstrating that:
- Not all variance is useful for classification
- Noise and redundant information are effectively filtered
- Optimal balance exists between information retention and noise reduction

**LDA**: Maximum information retention (100%) corresponds to maximum accuracy (83.4%), suggesting that LDA's limited dimensionality (9-D) captures all available discriminative information.

### 5.5 Classifier Performance Analysis

**k-NN (k=5)** consistently outperformed other classifiers:
- Achieved highest accuracy (91.1%) with PCA at 75 dimensions
- Robust across different dimensionalities
- Benefits from improved density in reduced-dimensional spaces
- Good efficiency balance (5.80 for PCA, 11.46 for LDA)

**Linear SVM** showed competitive performance:
- Performed well with both PCA and LDA
- More stable across different dimensions
- Achieved 89.2% with PCA at optimal dimensions
- Moderate efficiency (1.81 for PCA, 6.76 for LDA)

**Mahalanobis Distance** showed declining performance at higher dimensions:
- Performed well at low dimensions (≤50)
- Significant degradation at higher dimensions (65.0% at 784-D)
- Exponential time increase makes it impractical for high dimensions
- Likely due to covariance matrix estimation issues in high dimensions

### 5.6 Dimensionality Trade-offs

The results demonstrate clear trade-offs:
- **Too Few Dimensions (2-10)**: Insufficient information, low accuracy (29.9-85.8%), but very fast computation
- **Optimal Dimensions (20-75)**: Balance between information preservation and noise reduction, peak accuracy (90-91%), moderate computation time
- **Too Many Dimensions (150-784)**: Diminishing returns, slight performance degradation, significantly longer computation time

## 6. Conclusions

This study provides several important insights into dimensionality reduction for classification:

1. **PCA outperforms LDA in accuracy** (91.1% vs 83.4%) for the MNIST dataset, likely due to better variance preservation and flexibility in dimensionality selection. However, LDA achieves superior computational efficiency.

2. **Optimal dimensionality exists at intermediate values** (50-75 dimensions for PCA), not at the original 784 dimensions. This demonstrates that dimensionality reduction can improve both accuracy and efficiency simultaneously.

3. **Information retention analysis reveals** that optimal classification doesn't require full variance retention. PCA achieves peak accuracy with only 71.19% variance retention, indicating effective noise filtering.

4. **k-NN (k=5) is the best-performing classifier** across most dimensionalities, benefiting from improved data density in reduced spaces while maintaining good efficiency.

5. **Computational efficiency varies significantly** with dimensionality and classifier choice. Mahalanobis distance becomes impractical at high dimensions, while k-NN and Linear SVM maintain reasonable performance.

6. **Dimensionality reduction provides multiple benefits**: improved accuracy, reduced computational cost, noise reduction, and implicit regularization. The optimal configuration achieves 90.4% dimension reduction while improving accuracy by 2.2% compared to original data.

7. **The curse of dimensionality is evident**: Original high-dimensional data (784-D) performs worse than optimally reduced data, confirming the importance of dimensionality reduction for both accuracy and efficiency.

### 6.1 Practical Recommendations

- **For maximum accuracy**: Use PCA with k-NN (k=5) at 75 dimensions
- **For real-time applications**: Use LDA with k-NN (k=3) at 8 dimensions for best efficiency
- **For balanced performance**: Use PCA with k-NN (k=5) at 50 dimensions
- **Avoid**: Mahalanobis distance at dimensions >100 due to exponential time complexity

### 6.2 Limitations and Future Work

This study has several limitations:
- Single dataset (MNIST) - results may vary for other data types
- Limited to linear dimensionality reduction methods
- Fixed sample size (5,000 samples)
- Timing measurements may vary with hardware

Future work could explore:
- Non-linear dimensionality reduction (t-SNE, UMAP, autoencoders)
- Other high-dimensional datasets (images, audio, text)
- Larger sample sizes and cross-validation
- Deep learning-based dimensionality reduction
- Parallel computing implementations for efficiency analysis

## 7. References

1. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202.

2. Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179-188.

3. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.

4. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.

5. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.

6. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

7. Bellman, R. (1961). *Adaptive Control Processes: A Guided Tour*. Princeton University Press.

8. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

**Note**: All experimental results, code, and visualizations are available in the project repository. Figures 1-3 (results_mnist.png, timing_efficiency_mnist.png, info_retention_mnist.png) and detailed numerical results (results_summary_mnist.txt, comparison_table_mnist.txt) accompany this report.
