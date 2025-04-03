import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=3, metric='euclidean'):
        """
    k-Nearest Neighbors (KNN) Classifier written by Ali Özyüksel.

    Parameters:
    ------------
    k : int, default=3
        Number of nearest neighbors. Determines how many neighbors are considered when making a prediction. Odd numbers are preferred to avoid ties.

    metric : str, default="euclidean"
        The distance metric to use. It can be euclidean, manhattan or mahalanobis.

    Attributes:
    ----------
    X_train : ndarray
        Training data features.
    
    y_train : ndarray
        Training data labels.

    Methods:
    --------
    
    predict(X_test):
        Makes predictions for the given test data.

    accuracy(x_test, y_test):
        Computes the model's accuracy with respect to the formula accuracy = correct predictions / total predictions.
        

    confusion_matrix(x_test, y_test):
        Generates a confusion matrix based on the model's predictions.

    plotting(x_test, y_test):
        Visualizes the confusion matrix as a heatmap.

    """    
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    # distance functions.
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def mahalanobis_distance(self, x1, x2):
        diff = x1 - x2
        cov_matrix =np.cov(self.X_train, rowvar=False)  
        cov_matrix_inv = np.linalg.inv(cov_matrix)
        return np.sqrt(np.dot(np.dot(diff.T, cov_matrix_inv), diff))

    def compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
        elif self.metric == 'mahalanobis':
            return self.mahalanobis_distance(x1, x2)
        else:
            raise ValueError("Unsupported metric. Use 'euclidean', 'mahalanobis' or 'manhattan'.")

    # prediction function. it is basically a loop that iterates over the test data and finds the k nearest neighbors by cheking the distance.
    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            distances = [self.compute_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common = unique_labels[np.argmax(counts)]
            predictions.append(most_common)
        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        return accuracy

    #self written confusion matrix function.
    def confusion_matrix(self, x_test, y_test):
        y_pred = self.predict(x_test)
        unique_labels = np.unique(y_pred)
        num_classes = len(np.unique(y_test))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        for true_label, predicted_label in zip(y_test, y_pred):
            cm[label_to_index[true_label], label_to_index[predicted_label]] += 1
        return cm
    
    #plotting confusion matrix as a heatmap to visualize the performance of the model.
    def plotting(self, x_test, y_test):
        cm = self.confusion_matrix(x_test, y_test)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        
        plt.title(f"Confusion Matrix for k={self.k}, metric={self.metric}, accuracy={self.evaluate(x_test, y_test):.3f}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
