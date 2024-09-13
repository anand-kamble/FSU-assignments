# %% ====== Q1. A ======
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# %%
# Load the dataset using numpy
file_path = 'abalone.csv'
abalone_data: np.ndarray[np.any, np.dtype[np.floating[np.float64]]] = np.genfromtxt(
    file_path, delimiter=',')
# %%
# Split the data into predictors (first 7 columns) and response (8th column)
X: np.ndarray[np.any, np.dtype[np.floating[np.floating[np.float64]]]
              ] = abalone_data[:, :7]  # First 7 columns are predictors
y: np.ndarray[np.any, np.dtype[np.floating[np.floating[np.float64]]]
              ] = abalone_data[:, 7]   # 8th column is the response
# %%
# Number of splits
n_splits = 10

# Initialize lists to store MSE results for each split
train_mse: list = []
test_mse: list = []
# %%
# Perform 10 random splits
for _ in range(n_splits):
    # Split the data into 90% train and 10% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1)

    # Null model prediction: mean of the training set's y values
    y_train_mean = np.mean(y_train)

    # Predict mean for both train and test sets
    y_train_pred = np.full_like(y_train, y_train_mean)
    y_test_pred = np.full_like(y_test, y_train_mean)

    # Compute MSE for train and test sets
    _train_mse: np.float32 | np.ndarray = mean_squared_error(
        y_train, y_train_pred)
    _test_mse: np.float32 | np.ndarray = mean_squared_error(
        y_test, y_test_pred)

    # Append the results
    train_mse.append(_train_mse)
    test_mse.append(_test_mse)
# %%
# Compute average train and test MSE over the 10 splits
avg_train_mse = np.mean(train_mse)
avg_test_mse = np.mean(test_mse)

# Output the results
print(f"Average Training MSE: {avg_train_mse}")
print(f"Average Testing MSE: {avg_test_mse}")


# %% ====== Q1. B ======

# Regularization parameter
lambda_reg = 0.01

# Initialize lists to store results
train_r2 = []
test_r2 = []
train_mse = []
test_mse = []
log_determinant = []

# Perform 10 random splits
for _ in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    XtX = np.dot(X_train.T, X_train)
    Xty = np.dot(X_train.T, y_train)
    I: np.ndarray[np.any, np.dtype[np.floating[np.float64]]
                  ] = np.eye(X_train.shape[1])
    XtX_reg = XtX + lambda_reg * I
    beta = np.linalg.solve(XtX_reg, Xty)

    y_train_pred = np.dot(X_train, beta)
    y_test_pred = np.dot(X_test, beta)
    _train_mse = mean_squared_error(y_train, y_train_pred)
    _test_mse = mean_squared_error(y_test, y_test_pred)

    _train_r2: np.float32 | np.ndarray = r2_score(y_train, y_train_pred)
    _test_r2: np.float32 | np.ndarray = r2_score(y_test, y_test_pred)
    determinant = np.linalg.det(XtX_reg)

    train_r2.append(_train_r2)
    test_r2.append(_test_r2)
    train_mse.append(_train_mse)
    test_mse.append(_test_mse)
    log_determinant.append(np.log(determinant))

# Output the results
print(f"Average Training R2: {np.mean(train_r2)} (Std: {np.std(train_r2)})")
print(f"Average Test R2: {np.mean(test_r2)} (Std: {np.std(test_r2)})")
print(f"Average Training MSE: {np.mean(train_mse)} (Std: {np.std(train_mse)})")
print(f"Average Test MSE: {np.mean(test_mse)} (Std: {np.std(test_mse)})")
print(
    f"Average Log Determinant: {np.mean(log_determinant)} (Std: {np.std(log_determinant)})")

# %% ====== Q1. C ======


d = range(1, 8)
train_r2_avg, test_r2_avg = [], []
train_mse_avg, test_mse_avg = [], []

for max_depth in d:
    train_r2, test_r2 = [], []
    train_mse, test_mse = [], []

    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1)

        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(X_train, y_train)

        y_train_pred: np.ndarray = tree.predict(X_train)
        y_test_pred: np.ndarray = tree.predict(X_test)

        _train_r2 = r2_score(y_train, y_train_pred)
        _test_r2 = r2_score(y_test, y_test_pred)
        _train_mse = mean_squared_error(y_train, y_train_pred)
        _test_mse = mean_squared_error(y_test, y_test_pred)

        train_r2.append(_train_r2)
        test_r2.append(_test_r2)
        train_mse.append(_train_mse)
        test_mse.append(_test_mse)

    train_r2_avg.append(np.mean(train_r2))
    test_r2_avg.append(np.mean(test_r2))
    train_mse_avg.append(np.mean(train_mse))
    test_mse_avg.append(np.mean(test_mse))

# Plot average R2 vs tree depth
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(d, train_r2_avg, label='Train R2', marker='o')
plt.plot(d, test_r2_avg, label='Test R2', marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('Average R2')
plt.title('Average R2 vs Tree Depth')
plt.legend()
plt.grid(True)
plt.show()

# Plot average MSE vs tree depth
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(d, train_mse_avg, label='Train MSE', marker='o')
plt.plot(d, test_mse_avg, label='Test MSE', marker='o')
plt.axhline(y=avg_train_mse, color='r',
            linestyle='--', label='Null Model Train MSE')
plt.axhline(y=avg_test_mse, color='orange',
            linestyle='--', label='Null Model Test MSE')
plt.xlabel('Tree Depth')
plt.ylabel('Average MSE')
plt.title('Average MSE vs Tree Depth')
plt.legend()
plt.grid(True)
plt.show()

# %% ====== Q1. D ======

from sklearn.ensemble import RandomForestRegressor

# Define the number of trees
n_trees: list[int] = [10, 30, 100, 300]
train_r2_avg_list, oob_r2_avg_list, test_r2_avg_list = [], [], []
train_mse_avg_list, oob_mse_avg_list, test_mse_avg_list = [], [], []
train_r2_std_list, oob_r2_std_list, test_r2_std_list = [], [], []
train_mse_std_list, oob_mse_std_list, test_mse_std_list = [], [], []

# Iterate over different tree numbers
for n_tree in n_trees:
    train_r2_list, oob_r2_list, test_r2_list = [], [], []
    train_mse_list, oob_mse_list, test_mse_list = [], [], []

    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # Train random forest with OOB enabled
        forest = RandomForestRegressor(n_estimators=n_tree, oob_score=True, random_state=None)
        forest.fit(X_train, y_train)

        # Predictions for training and test sets
        y_train_pred = forest.predict(X_train)
        y_test_pred = forest.predict(X_test)

        # Training and test R2 and MSE
        _train_r2 = r2_score(y_train, y_train_pred)
        _test_r2 = r2_score(y_test, y_test_pred)
        _train_mse = mean_squared_error(y_train, y_train_pred)
        _test_mse = mean_squared_error(y_test, y_test_pred)

        # OOB R2 and MSE
        _oob_r2 = forest.oob_score_
        oob_preds = forest.oob_prediction_
        _oob_mse = mean_squared_error(y_train, oob_preds)

        # Append the results
        train_r2_list.append(_train_r2)
        test_r2_list.append(_test_r2)
        oob_r2_list.append(_oob_r2)
        train_mse_list.append(_train_mse)
        test_mse_list.append(_test_mse)
        oob_mse_list.append(_oob_mse)

    # Append averages and standard deviations
    train_r2_avg_list.append(np.mean(train_r2_list))
    test_r2_avg_list.append(np.mean(test_r2_list))
    oob_r2_avg_list.append(np.mean(oob_r2_list))
    train_mse_avg_list.append(np.mean(train_mse_list))
    test_mse_avg_list.append(np.mean(test_mse_list))
    oob_mse_avg_list.append(np.mean(oob_mse_list))

    train_r2_std_list.append(np.std(train_r2_list))
    test_r2_std_list.append(np.std(test_r2_list))
    oob_r2_std_list.append(np.std(oob_r2_list))
    train_mse_std_list.append(np.std(train_mse_list))
    test_mse_std_list.append(np.std(test_mse_list))
    oob_mse_std_list.append(np.std(oob_mse_list))

# Output results
for i, n_tree in enumerate(n_trees):
    print(f"Random Forest with {n_tree} trees:")
    print(f"Average Train R2: {train_r2_avg_list[i]:.4f} ± {train_r2_std_list[i]:.4f}")
    print(f"Average OOB R2: {oob_r2_avg_list[i]:.4f} ± {oob_r2_std_list[i]:.4f}")
    print(f"Average Test R2: {test_r2_avg_list[i]:.4f} ± {test_r2_std_list[i]:.4f}")
    print(f"Average Train MSE: {train_mse_avg_list[i]:.4f} ± {train_mse_std_list[i]:.4f}")
    print(f"Average OOB MSE: {oob_mse_avg_list[i]:.4f} ± {oob_mse_std_list[i]:.4f}")
    print(f"Average Test MSE: {test_mse_avg_list[i]:.4f} ± {test_mse_std_list[i]:.4f}")
    print("\n")


# %%
