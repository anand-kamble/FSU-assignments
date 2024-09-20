"""
Author: Anand Kamble
Course: Machine Learning (STA5635)
Assignment: Homework 03
Date: 19th Sep 2024
email: amk23j@fsu.edu
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#%%
# Hinge loss function
def hinge_loss(w, X, y):
    return np.maximum(0, 1 - y * np.dot(X, w))
#%%
# Gradient of the hinge loss
def gradient_hinge_loss(w, X, y, lambda_):
    N = len(y)
    grad = np.zeros_like(w)
    for i in range(N):
        if y[i] * np.dot(X[i], w) < 1:
            grad += -y[i] * X[i]
    grad /= N
    grad += 2 * lambda_ * w
    return grad
#%%
# Gradient descent
def gradient_descent(X, y, lambda_, eta, num_iter):
    N, d = X.shape
    w = np.zeros(d)
    losses = []
    
    for it in range(num_iter):
        loss = np.mean(hinge_loss(w, X, y)) + lambda_ * np.dot(w, w)
        losses.append(loss)
        
        # Update weights
        grad = gradient_hinge_loss(w, X, y, lambda_)
        w -= eta * grad
        
        # Print the loss every 50 iterations for monitoring
        if it % 50 == 0:
            print(f"Iteration {it}: Loss = {loss}")
    
    return w, losses
#%%
# Normalize data
def normalize_data(X_train, X_test):
    scaler = StandardScaler()   
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
#%%
# Misclassification error
def misclassification_error(X, y, w):
    preds = np.sign(np.dot(X, w))
    return 1 - accuracy_score(y, preds)
#%%
# ROC curve
def plot_roc_curve(X, y, w, label):
    preds = np.dot(X, w)
    fpr, tpr, _ = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

#%%
# Load your Gisette dataset here
train_data_path = '/home/amk23j/Documents/FSU-assignments/Machine Learning/HW03/Gisette/gisette_train.data'
train_labels_path = '/home/amk23j/Documents/FSU-assignments/Machine Learning/HW03/Gisette/gisette_train.labels'
test_data_path = '/home/amk23j/Documents/FSU-assignments/Machine Learning/HW03/Gisette/gisette_valid.data'
test_labels_path = '/home/amk23j/Documents/FSU-assignments/Machine Learning/HW03/Gisette/gisette_valid.labels'

X_train = np.loadtxt(train_data_path)
y_train = np.loadtxt(train_labels_path)
X_test = np.loadtxt(test_data_path)
y_test = np.loadtxt(test_labels_path)


#%%
# Hyperparameters
lambda_ = 0.001
eta = 0.01  # You may need to tune this
num_iter = 300
#%%
# Normalize the data
X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)

# Run gradient descent
w, losses = gradient_descent(X_train_scaled, y_train, lambda_, eta, num_iter)
#%%
# Plot training loss vs iterations
plt.figure()
plt.plot(range(num_iter), losses)
plt.xlabel('Iterations')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Iterations (Dexter)')
# plt.show()
#%%
# Report misclassification error
train_error = misclassification_error(X_train_scaled, y_train, w)
test_error = misclassification_error(X_test_scaled, y_test, w)
print(f'Training misclassification error: {train_error}')
print(f'Test misclassification error: {test_error}')
#%%
# Plot ROC curve for training and test set
# plt.figure()
# plot_roc_curve(X_train_scaled, y_train, w, 'Train')
# plot_roc_curve(X_test_scaled, y_test, w, 'Test')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve (Dexter)')
# plt.legend(loc='best')
# plt.show()

# %%
# PART D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

train_data_path = '/home/amk23j/Documents/FSU-assignments/Machine Learning/HW03/Gisette/gisette_train.data'
train_labels_path = '/home/amk23j/Documents/FSU-assignments/Machine Learning/HW03/Gisette/gisette_train.labels'
test_data_path = '/home/amk23j/Documents/FSU-assignments/Machine Learning/HW03/Gisette/gisette_valid.data'
test_labels_path = '/home/amk23j/Documents/FSU-assignments/Machine Learning/HW03/Gisette/gisette_valid.labels'

X_train = np.loadtxt(train_data_path)
y_train = np.loadtxt(train_labels_path)
X_test = np.loadtxt(test_data_path)
y_test = np.loadtxt(test_labels_path)

# Normalize the variables of the training set to have zero mean and standard deviation 1
mean_X_train = np.mean(X_train, axis=0)
std_X_train = np.std(X_train, axis=0)
# Avoid division by zero for features with zero std
std_X_train[std_X_train == 0] = 1

X_train_norm = (X_train - mean_X_train) / std_X_train
X_test_norm = (X_test - mean_X_train) / std_X_train  # Use training mean and std

# Initialize w(0) = 0
w = np.zeros(X_train_norm.shape[1])

# Set regularization parameter λ = 0.001
lambd = 0.001

# Set h = 0.1 for Huberized SVM loss
h = 0.1

# Define the Huberized SVM loss function ℓ(s) and its derivative ℓ'(s)
def huberized_svm_loss(s, h):
    loss = np.where(s >= 1,
                    0,
                    np.where(s <= 1 - h,
                             1 - s - h/2,
                             (1 - s)**2 / (2 * h)))
    return loss

def huberized_svm_loss_derivative(s, h):
    derivative = np.where(s >= 1,
                          0,
                          np.where(s <= 1 - h,
                                   -1,
                                   - (1 - s) / h))
    return derivative

# Function to compute loss and gradient
def compute_loss_and_gradient(w, X, y, lambd, h):
    N = X.shape[0]
    s = y * (X @ w)  # s_i = y_i * (w^T x_i)
    loss_i = huberized_svm_loss(s, h)
    loss = np.mean(loss_i) + lambd * np.dot(w, w)
    # Compute derivative of loss function
    l_prime = huberized_svm_loss_derivative(s, h)
    # Gradient: ∂L/∂w = (1/N) ∑ [ℓ'(s_i) y_i x_i ] + 2 λ w
    grad = (1 / N) * np.sum((l_prime * y)[:, np.newaxis] * X, axis=0) + 2 * lambd * w
    return loss, grad

# Gradient descent parameters
num_iterations = 300
eta = 0.01  # Learning rate (adjusted to ensure convergence within 300 iterations)

loss_history = []

for iteration in range(num_iterations):
    loss, grad = compute_loss_and_gradient(w, X_train_norm, y_train, lambd, h)
    loss_history.append(loss)
    w = w - eta * grad
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Loss: {loss}")

# Compute misclassification error on the training set
scores_train = X_train_norm @ w
y_pred_train = np.sign(scores_train)
error_rate_train = np.mean(y_pred_train != y_train)
print(f"Training misclassification error: {error_rate_train}")

# Compute misclassification error on the test set
scores_test = X_test_norm @ w
y_pred_test = np.sign(scores_test)
error_rate_test = np.mean(y_pred_test != y_test)
print(f"Test misclassification error: {error_rate_test}")

# Plot the training loss vs iteration number
# plt.figure()
#%%
plt.plot(range(num_iterations), loss_history, label='Huberized SVM Loss')
plt.plot(range(num_iterations), losses, label='Hinge Loss')
plt.xlabel('Iteration Number')
plt.ylabel('Training Loss')
plt.legend()
plt.title('Training Loss vs Iteration Number')
plt.show()
#%%
# Compute ROC curves
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, scores_train)
roc_auc_train = auc(fpr_train, tpr_train)

fpr_test, tpr_test, thresholds_test = roc_curve(y_test, scores_test)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC curves
plt.figure()
plt.plot(fpr_train, tpr_train, label='Training ROC (AUC = %0.2f)' % roc_auc_train)
plt.plot(fpr_test, tpr_test, label='Test ROC (AUC = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# %%
