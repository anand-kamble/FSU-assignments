# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from typing import Literal

y_train = None
X_train = None
X_test = None
y_test = None

DATASET: Literal['Gisette', 'dexter', 'Madelon'] = 'Madelon'
match DATASET:
    case 'Gisette':
        train_data_path = './Gisette/gisette_train.data'
        train_labels_path = './Gisette/gisette_train.labels'
        test_data_path = './Gisette/gisette_valid.data'
        test_labels_path = './Gisette/gisette_valid.labels'

        X_train = np.loadtxt(train_data_path)
        y_train = np.loadtxt(train_labels_path)
        X_test = np.loadtxt(test_data_path)
        y_test = np.loadtxt(test_labels_path)

    case 'dexter':
        train_data_path = './dexter/dexter_train.csv'
        train_labels_path = './dexter/dexter_train.labels'
        test_data_path = './dexter/dexter_valid.csv'
        test_labels_path = './dexter/dexter_valid.labels'

        X_train = np.loadtxt(train_data_path, delimiter=',')
        y_train = np.loadtxt(train_labels_path)
        X_test = np.loadtxt(test_data_path, delimiter=',')
        y_test = np.loadtxt(test_labels_path)

    case 'Madelon':
        train_data_path = './MADELON/madelon_train.data'
        train_labels_path = './MADELON/madelon_train.labels'
        test_data_path = './MADELON/madelon_valid.data'
        test_labels_path = './MADELON/madelon_valid.labels'

        X_train = np.loadtxt(train_data_path)
        y_train = np.loadtxt(train_labels_path)
        X_test = np.loadtxt(test_data_path)
        y_test = np.loadtxt(test_labels_path)

# %%
# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use the same scaler for test data
# %%
# Define the hard-thresholding operator


def hard_threshold(w, lambd):
    w_thresholded = w.copy()
    w_thresholded[np.abs(w_thresholded) < lambd] = 0
    return w_thresholded

# Gradient of squared hinge loss


def gradient_squared_hinge_loss(X, y, w):
    n = X.shape[0]
    margins = y * (X.dot(w))
    indicator = (margins < 1).astype(float)
    temp = -2 * y * (1 - margins) * indicator
    grad = X.T.dot(temp) / n
    return grad

# Implement the TISP algorithm


def TISP(X, y, w0, alpha, lambd, num_iter):
    w = w0.copy()
    history = []
    for t in range(num_iter):
        grad = gradient_squared_hinge_loss(X, y, w)
        w = w - alpha * grad
        w = hard_threshold(w, lambd)
        history.append(w.copy())
    return w, history


# %%
# Initialize parameters
n_features = X_train_scaled.shape[1]
w0 = np.zeros(n_features)
alpha = 0.01  # Step size (you may need to adjust this)
num_iter = 200  # Number of iterations

lambdas: list[float] = []
match DATASET:
    case 'Gisette':
        lambdas = [0.0108, 0.008, 0.0048, 0.0028, 0.0022]

    case 'dexter':
        lambdas = [0.006, 0.00405, 0.0029, 0.00216, 0.0019]

    case 'Madelon':
        lambdas = [0.0019, 0.00095, 0.000699, 0.000303, 0.000001]


# Lists to store results
feature_counts = []
train_errors = []
test_errors = []
lambda_list = []

# Run TISP for each lambda and record the number of selected features and errors
for lambd in lambdas:
    w, history = TISP(X_train_scaled, y_train, w0, alpha, lambd, num_iter)
    num_features = np.sum(w != 0)
    feature_counts.append(num_features)
    lambda_list.append(lambd)

    # Compute train misclassification error
    y_train_pred = np.sign(X_train_scaled.dot(w))
    train_error = np.mean(y_train_pred != y_train)
    train_errors.append(train_error)

    # Compute test misclassification error
    y_test_pred = np.sign(X_test_scaled.dot(w))
    test_error = np.mean(y_test_pred != y_test)
    test_errors.append(test_error)

# Create a DataFrame to store results
results = pd.DataFrame({
    'lambda': lambda_list,
    'num_features': feature_counts,
    'train_error': train_errors,
    'test_error': test_errors
})

# Sort results by the number of features
results = results.sort_values('num_features')


# %%
# Desired numbers of features
desired_features = [10, 30, 100, 300, 500]

# Find lambdas corresponding to desired numbers of features
selected_entries = []
for k in desired_features:
    idx = (results['num_features'] - k).abs().argmin()
    row = results.iloc[idx]
    selected_entries.append({
        'lambda': row['lambda'],
        'num_features': int(row['num_features']),
        'train_error': row['train_error'],
        'test_error': row['test_error']
    })

# Create a table of the selected entries
table_df = pd.DataFrame(selected_entries)
# %%
# Plot the train misclassification error vs iteration number for 300 features
# Find the lambda corresponding to approximately 300 features
lambda_300 = table_df.loc[table_df['num_features'] == 300, 'lambda'].values
if len(lambda_300) == 0:
    # If exact 300 features is not found, select the closest one
    idx = (table_df['num_features'] - 300).abs().argmin()
    lambda_300 = table_df.iloc[idx]['lambda']
else:
    lambda_300 = lambda_300[0]

# Run TISP with lambda_300 and record train error at each iteration
w0 = np.zeros(n_features)
lambd = lambda_300
w, history = TISP(X_train_scaled, y_train, w0, alpha, lambd, num_iter)

train_errors_iter = []
for w_t in history:
    y_train_pred = np.sign(X_train_scaled.dot(w_t))
    train_error = np.mean(y_train_pred != y_train)
    train_errors_iter.append(train_error)

# Plotting train misclassification error vs iteration number
plt.figure()
plt.plot(range(1, num_iter + 1), train_errors_iter)
plt.xlabel('Iteration Number')
plt.ylabel('Train Misclassification Error')
plt.title(
    f'Train Misclassification Error vs Iteration Number ({np.sum(w != 0)} features)')
plt.grid(True)
plt.savefig(f"{DATASET}_trainerror vs iter num.png")
# %%
# Plot final train and test misclassification error vs number of selected features
plt.figure()
plt.plot(results['num_features'], results['train_error'], label='Train Error')
plt.plot(results['num_features'], results['test_error'], label='Test Error')
plt.xlabel('Number of Selected Features')
plt.ylabel('Misclassification Error')
plt.title('Misclassification Error vs Number of Selected Features')
plt.legend()
plt.grid(True)
plt.savefig(f"{DATASET}_Misclassification Error vs Number of Selected Features.png")
# %%
# Plot ROC curves for the model with 300 features
# Compute scores for train and test data
w_300 = w  # We already have w from the TISP run above
y_train_scores = X_train_scaled.dot(w_300)
y_test_scores = X_test_scaled.dot(w_300)

# Compute ROC curve and AUC for train set
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_scores)
roc_auc_train = auc(fpr_train, tpr_train)

# Compute ROC curve and AUC for test set
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_scores)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC curves
plt.figure()
plt.plot(fpr_train, tpr_train, label='Train ROC (AUC = %0.2f)' % roc_auc_train)
plt.plot(fpr_test, tpr_test, label='Test ROC (AUC = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curves for Model with {np.sum(w != 0)} Features')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(f'{DATASET}_ROC Curves for Model.png')

# %%
print("Results:")
print(table_df)
print("Results saved in the current directory for the dataset:", DATASET)