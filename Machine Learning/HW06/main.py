#%%
import numpy as np
from sklearn.metrics import log_loss, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import multiprocessing as mp
from functools import partial

sys.path.append('/home/amk23j/Documents/FSU-assignments/Machine Learning/')
from dataset import load_dataset

class LogitBoost:
    def __init__(self, num_iterations=100, num_bins=10):
        self.num_iterations = num_iterations
        self.num_bins = num_bins
        self.weak_learners = []

    def _fit_weak_learner(self, X, z, w):
        best_feature = None
        best_bin_means = None
        best_loss = float('inf')

        for feature in range(X.shape[1]):
            bins = np.linspace(X[:, feature].min(), X[:, feature].max(), self.num_bins)
            bin_indices = np.digitize(X[:, feature], bins)
            bin_means = np.zeros(self.num_bins)

            for b in range(1, self.num_bins + 1):
                mask = bin_indices == b
                if np.sum(mask) > 0:
                    bin_means[b - 1] = np.sum(w[mask] * z[mask]) / np.sum(w[mask])

            predictions = bin_means[bin_indices - 1]
            loss = np.sum(w * (z - predictions) ** 2)

            if loss < best_loss:
                best_loss = loss
                best_feature = feature
                best_bin_means = bin_means

        return best_feature, best_bin_means

    def fit(self, X, y):
        N = X.shape[0]
        h = np.zeros(N)  # Initial classifier
        self.loss_history = []

        for iteration in tqdm(range(self.num_iterations), desc="Fitting LogitBoost"):
            z = y / (1 + np.exp(y * h))
            w = np.exp(y * h) / (1 + np.exp(y * h)) ** 2

            feature, bin_means = self._fit_weak_learner(X, z, w)
            bin_indices = np.digitize(X[:, feature], np.linspace(X[:, feature].min(), X[:, feature].max(), self.num_bins))
            h_new = bin_means[bin_indices - 1]
            h += h_new

            loss = np.sum(np.log(1 + np.exp(-y * h)))
            self.loss_history.append(loss)

            self.weak_learners.append((feature, bin_means))

    def predict(self, X):
        N = X.shape[0]
        h = np.zeros(N)

        for feature, bin_means in self.weak_learners:
            bin_indices = np.digitize(X[:, feature], np.linspace(X[:, feature].min(), X[:, feature].max(), self.num_bins))
            h += bin_means[bin_indices - 1]
        
        return np.sign(h), h

def train_and_evaluate(k, X_train, y_train, X_test, y_test):
    print(f"Training for k={k}")
    logitboost = LogitBoost(num_iterations=k)
    logitboost.fit(X_train, y_train)
    
    y_train_pred, _ = logitboost.predict(X_train)
    y_test_pred, _ = logitboost.predict(X_test)
    
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)
    
    return k, train_error, test_error, logitboost.loss_history, logitboost

#%%
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_dataset("Gisette")


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # y_train = 2 * y_train - 1
    # y_test = 2 * y_test - 1
    
    print(np.unique(y_train))  # Check if y_train contains only {-1, 1}
    print(np.unique(y_test))   # Check if y_test contains only {-1, 1}
    
    ks = [10, 30, 100, 300, 500]

    with mp.Pool() as pool:
        results = pool.map(partial(train_and_evaluate, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test), ks)

    train_errors, test_errors, loss_histories = [], [], []
    logitboost_300 = None

    for k, train_error, test_error, loss_history, logitboost in results:
        train_errors.append(train_error)
        test_errors.append(test_error)
        loss_histories.append(loss_history)
        if k == 300:
            logitboost_300 = logitboost

    # Plotting code remains the same
    plt.figure()
    plt.plot(loss_histories[4], label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Iterations (k=500)')
    plt.legend()
    plt.show()
    loss_data = np.array(loss_histories[4])
    np.savetxt('loss_history_k500.dat', np.column_stack((np.arange(len(loss_data)), loss_data)), fmt='%.6f', header='Iteration Loss', comments='')


    plt.figure()
    plt.plot(ks, train_errors, label='Training Error', marker='o')
    plt.plot(ks, test_errors, label='Test Error', marker='o')
    plt.xlabel('Number of Iterations (k)')
    plt.ylabel('Misclassification Error')
    plt.title('Misclassification Error vs k')
    plt.legend()
    plt.show()
    error_data = np.column_stack((ks, train_errors, test_errors))
    np.savetxt('errors_vs_k.dat', error_data, fmt='%.6f', header='k Train_Error Test_Error', comments='')




    _, y_train_scores = logitboost_300.predict(X_train)
    _, y_test_scores = logitboost_300.predict(X_test)

    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_scores)
    roc_auc_train = auc(fpr_train, tpr_train)

    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_scores)
    roc_auc_test = auc(fpr_test, tpr_test)

    plt.figure()
    plt.plot(fpr_train, tpr_train, label=f'Training ROC (AUC = {roc_auc_train:.2f})')
    plt.plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {roc_auc_test:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for k=300')
    plt.legend()
    plt.show()
    
    roc_train_data = np.column_stack((fpr_train, tpr_train))
    np.savetxt('roc_train_k300.dat', roc_train_data, fmt='%.6f', header='FPR TPR', comments='')

    roc_test_data = np.column_stack((fpr_test, tpr_test))
    np.savetxt('roc_test_k300.dat', roc_test_data, fmt='%.6f', header='FPR TPR', comments='')

# %%
