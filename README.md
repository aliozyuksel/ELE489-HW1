# K-Nearest Neighbors (KNN) Classification

This repository contains an implementation of the K-Nearest Neighbors (KNN) algorithm from scratch. The project evaluates different distance metrics, including Euclidean, Manhattan, and Mahalanobis, using the Wine dataset.

There are also accuracy evluation, confusion matrix visualization, and accuracy versus k-value graph as well as classification report for each metric and k value.

## Files

- `analysis.ipynb`: Jupyter Notebook that contains data analysis, KNN implementation, and evaluation.
- `knn.py`: Python script implementing the KNN algorithm.
- `wine.data`: Dataset used for training and testing.

## Data
 
- There are 13 features and 3 labels in the data.
- 178 sample presents.
## How to run

- To run this project, put wine.data, analysis.ipynb and knn.py in the same folder.
- Thereafter, each block in the analysis.ipynb can be run. There is no need to run knn.py.

## Requirements

To run this project, install the required libraries using:

- matplotlib
- seaborn
- pandas
- numpy
- IPython
- scikit-learn

You can install them by just writing the command below to your command promt.

```bash
pip install matplotlib seaborn pandas numpy IPython scikit-learn
