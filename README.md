# Gene Expression Clustering using Unsupervised Machine Learning

## Overview
This project explores unsupervised machine learning to group biological samples
based on their measurements, without telling the algorithm the correct answer in
advance. The goal is to see whether natural patterns in the data align with known
disease categories — a core idea behind gene expression clustering used in real
bioinformatics research (e.g., discovering disease subtypes).

## Dataset
- **Source:** Breast Cancer Wisconsin Dataset (built into scikit-learn)
- **Samples:** 569
- **Features:** 30 measurements per sample (used here as stand-ins for gene
  expression-style data)
- **Note:** The true malignant/benign labels were hidden from the clustering
  algorithm and only used afterward to check the results.

## Approach
1. Loaded and scaled the data
2. Reduced 30 features to 2 dimensions using PCA, for visualization
3. Applied **K-Means Clustering** to group samples into 2 clusters
4. Applied **Hierarchical Clustering** as a second, independent method
5. Compared both sets of clusters against the real labels
6. Visualized results with a scatter plot and a dendrogram

## Key Finding
The unsupervised clusters closely aligned with the actual malignant/benign labels,
showing that meaningful biological groupings can emerge from the data itself,
without supervision — the same principle used to discover disease subtypes from
real gene expression data.

## Tools Used
- Python
- pandas
- scikit-learn
- matplotlib
- scipy

## How to Run
```bash
pip3 install pandas scikit-learn matplotlib scipy
python3 gene_expression_clustering.py
```

## Outputs
- `kmeans_clusters.png` — scatter plot of K-Means clusters
- `dendrogram.png` — tree diagram showing sample relationships from hierarchical clustering

## Future Improvements
- Apply to real gene expression data (e.g., from NCBI GEO)
- Try different numbers of clusters
- Compare additional clustering algorithms (DBSCAN, Gaussian Mixture Models)

## Note
This project was built with AI assistance for learning purposes — used to understand
clustering concepts and debug code, while making sure I understood the logic behind
every step.
