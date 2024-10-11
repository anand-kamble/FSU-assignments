#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
import time

import sys
sys.path.append('/home/amk23j/Documents/FSU-assignments/Machine Learning/')
from dataset import load_dataset

#%%
# Function to load the Gisette dataset
def load_gisette():
    # Replace 'gisette_train.data' and 'gisette_train.labels' with your actual file paths
    X_train, y_train = load_svmlight_file('gisette_train.data')
    X_train = X_train.toarray()
    y_train = np.loadtxt('gisette_train.labels')
    
    X_test, y_test = load_svmlight_file('gisette_valid.data')
    X_test = X_test.toarray()
    y_test = np.loadtxt('gisette_valid.labels')
    
    # Convert labels from {+1, -1} to {1, 0}
    y_train = (y_train + 1) / 2
    y_test = (y_test + 1) / 2
    
    return X_train, y_train, X_test, y_test
#%%
# LogitBoost implementation
class LogitBoost:
    def __init__(self, n_estimators=100, n_bins=10):
        self.n_estimators = n_estimators  # Number of boosting iterations
        self.n_bins = n_bins              # Number of bins for piecewise constant regressors
        self.models = []                  # List to store the selected weak learners
        self.feature_indices = []         # List to store selected feature indices

    def fit(self, X, y):
        N, D = X.shape
        # Initialize model output to zero
        self.F = np.zeros(N)
        # Convert labels to {-1, 1}
        y_tilde = 2 * y - 1

        # List to store loss at each iteration
        self.train_loss = []

        for m in range(self.n_estimators):
            start_time = time.time()
            # Compute probabilities
            p = 1 / (1 + np.exp(-self.F))
            # Compute pseudo-residuals
            w = p * (1 - p)
            z = (y_tilde - p) / w
            # Initialize variables to store the best feature and loss reduction
            best_loss = np.inf
            best_feature = None
            best_model = None
            # Iterate over all features
            for j in range(D):
                # Check the number of unique values in the feature
                unique_values = np.unique(X[:, j])
                if len(unique_values) < 2:
                    continue  # Skip features with a single unique value
                # Adjust number of bins if necessary
                n_bins_feature = min(self.n_bins, len(unique_values))
                # Discretize feature j into bins
                est = KBinsDiscretizer(n_bins=n_bins_feature, encode='onehot-dense', strategy='quantile')
                try:
                    X_binned = est.fit_transform(X[:, j].reshape(-1, 1))
                except ValueError as e:
                    # Skip features that cannot be binned properly
                    continue
                # Get the actual number of bins after transformation
                actual_bins = X_binned.shape[1]
                if actual_bins < 1:
                    continue  # Skip if no bins are created
                # Fit a piecewise constant regressor
                bin_predictions = np.zeros(actual_bins)
                for b in range(actual_bins):
                    idx = X_binned[:, b] == 1
                    if np.sum(w[idx]) == 0:
                        bin_predictions[b] = 0
                    else:
                        bin_predictions[b] = np.sum(w[idx] * z[idx]) / np.sum(w[idx])
                # Predict on training data
                y_pred = X_binned @ bin_predictions
                # Update model output
                F_new = self.F + y_pred
                # Compute new loss
                loss = np.sum(np.log(1 + np.exp(-y_tilde * F_new)))
                # Check if this feature gives a better loss
                if loss < best_loss:
                    best_loss = loss
                    best_feature = j
                    best_bin_predictions = bin_predictions.copy()
                    best_estimator = est
            # Update the model with the best weak learner
            if best_feature is None:
                print("No suitable feature found at iteration", m+1)
                break
            self.models.append((best_feature, best_estimator, best_bin_predictions))
            self.feature_indices.append(best_feature)
            # Update F
            X_binned_best = best_estimator.transform(X[:, best_feature].reshape(-1, 1))
            y_pred_best = X_binned_best @ best_bin_predictions
            self.F += y_pred_best
            # Store the training loss
            self.train_loss.append(best_loss)
            end_time = time.time()
            print(f"Iteration {m+1}/{self.n_estimators}, Feature {best_feature}, Loss: {best_loss:.4f}, Time: {end_time - start_time:.2f}s")


    
    def predict_proba(self, X):
        N = X.shape[0]
        F = np.zeros(N)
        # Sum the predictions from all weak learners
        for m in range(len(self.models)):
            feature, est, bin_preds = self.models[m]
            X_binned = est.transform(X[:, feature].reshape(-1, 1))
            F += X_binned @ bin_preds
        # Compute probabilities
        p = 1 / (1 + np.exp(-F))
        return p

    def predict(self, X):
        # Predict class labels
        p = self.predict_proba(X)
        return (p >= 0.5).astype(int)
#%%
# Main script
if __name__ == "__main__":
    # Load Gisette dataset
    X_train, y_train, X_test, y_test = load_dataset("Gisette")
    
    # Set the number of boosting iterations
    k_values = [10] #, 30, 100, 300, 500]
    train_errors = []
    test_errors = []
    
    for k in k_values:
        print(f"\nTraining LogitBoost with {k} iterations...")
        # Initialize LogitBoost classifier
        lb = LogitBoost(n_estimators=k, n_bins=10)
        # Fit the model
        lb.fit(X_train, y_train)
        # Predict on training and test sets
        y_train_pred = lb.predict(X_train)
        y_test_pred = lb.predict(X_test)
        # Compute misclassification errors
        train_error = 1 - accuracy_score(y_train, y_train_pred)
        test_error = 1 - accuracy_score(y_test, y_test_pred)
        train_errors.append(train_error)
        test_errors.append(test_error)
        print(f"Training Error: {train_error:.4f}, Test Error: {test_error:.4f}")
    
    # Plot training loss vs iteration number for k=500
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 501), lb.train_loss)
    plt.xlabel('Iteration Number')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Iteration Number (k=500)')
    plt.grid(True)
    plt.show()
    
    # Plot misclassification errors on training and test set vs k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_errors, label='Training Error', marker='o')
    plt.plot(k_values, test_errors, label='Test Error', marker='o')
    plt.xlabel('Number of Boosting Iterations (k)')
    plt.ylabel('Misclassification Error')
    plt.title('Misclassification Error vs Number of Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Report misclassification errors in a table
    print("\nMisclassification Errors:")
    print("k\tTraining Error\tTest Error")
    for i, k in enumerate(k_values):
        print(f"{k}\t{train_errors[i]:.4f}\t\t{test_errors[i]:.4f}")
    
    # Plot ROC curves for k=300
    print("\nPlotting ROC curves for k=300...")
    lb = LogitBoost(n_estimators=300, n_bins=10)
    lb.fit(X_train, y_train)
    y_train_scores = lb.predict_proba(X_train)
    y_test_scores = lb.predict_proba(X_test)
    # Compute ROC curves
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_scores)
    roc_auc_train = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_scores)
    roc_auc_test = auc(fpr_test, tpr_test)
    # Plot ROC curves
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_train, tpr_train, label=f'Training ROC curve (area = {roc_auc_train:.2f})')
    plt.plot(fpr_test, tpr_test, label=f'Test ROC curve (area = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (k=300)')
    plt.legend()
    plt.grid(True)
    plt.show()

# %%
