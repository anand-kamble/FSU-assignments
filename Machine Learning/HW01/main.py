# %%
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# %%
# Load the data

# MADELON dataset
# X_train: np.ndarray[np.float64] = np.loadtxt('./MADELON/madelon_train.data')
# X_test: np.ndarray[np.float64] = np.loadtxt('./MADELON/madelon_valid.data')
# Y_train: np.ndarray[np.float64] = np.loadtxt('./MADELON/madelon_train.labels')
# Y_test: np.ndarray[np.float64] = np.loadtxt('./MADELON/madelon_valid.labels')

# satimage dataset
X_train: np.ndarray[np.float64] = np.loadtxt('./satimage/X.dat')
X_test: np.ndarray[np.float64] = np.loadtxt('./satimage/Xtest.dat')
Y_train: np.ndarray[np.float64] = np.loadtxt('./satimage/Y.dat')
Y_test: np.ndarray[np.float64] = np.loadtxt('./satimage/Ytest.dat')

# %%
train_error: list = []
test_error: list = []
# %%
for i in range(12):
    tree = DecisionTreeClassifier(max_depth=i+1)
    tree.fit(X_train, Y_train)
    Y_pred_test: np.ndarray = tree.predict(X_test)
    test_error.append(1 - accuracy_score(Y_test, Y_pred_test))
    Y_pred_train: np.ndarray = tree.predict(X_train)
    train_error.append(1 - accuracy_score(Y_train, Y_pred_train))

# %%
min_test_error: float = min(test_error)
optimal_depth: int = test_error.index(min_test_error) + 1

print(f'Optimal depth: {optimal_depth}')
print(f'Minimum test error: {min_test_error}')

print(f"{'Depth':<10}{'Train Error':<15}{'Test Error':<15}")

# Loop through the data and print each row
for i, (train, test) in enumerate(zip(train_error, test_error), 1):
    print(f"{i:<10}{train:<15} {test:<15}")

# %%
plt.figure(figsize=(10, 6),dpi=300)
plt.xlabel('Tree Depth')
plt.ylabel('Misclassification Error')
plt.title('Training and Test Misclassification Errors vs Tree Depth')
plt.grid(True)
plt.plot(range(1,13),train_error, label='Training Error', marker='o')
plt.plot(range(1,13),test_error, label='Test Error', marker='o')
plt.legend()
plt.show()

#%%
# Part C

# MADELON dataset
X_train: np.ndarray[np.float64] = np.loadtxt('./MADELON/madelon_train.data')
X_test: np.ndarray[np.float64] = np.loadtxt('./MADELON/madelon_valid.data')
Y_train: np.ndarray[np.float64] = np.loadtxt('./MADELON/madelon_train.labels')
Y_test: np.ndarray[np.float64] = np.loadtxt('./MADELON/madelon_valid.labels')


#%%
k: list[int] = [3, 10, 30, 100, 300]

train_error: list = []
test_error: list = []

for k_value in k:
    forest = RandomForestClassifier(n_estimators=k_value, max_features=None)
    forest.fit(X_train, Y_train)
    
    Y_pred_test: np.ndarray = forest.predict(X_test)
    test_error.append(1 - accuracy_score(Y_test, Y_pred_test))
    
    Y_pred_train: np.ndarray = forest.predict(X_train)
    train_error.append(1 - accuracy_score(Y_train, Y_pred_train))
    

plt.figure(figsize=(10, 6),dpi=300)
plt.plot(k, train_error, label='Training Error', marker='o')
plt.plot(k, test_error, label='Test Error', marker='o')
plt.xlabel('Number of Trees (k)')
plt.ylabel('Misclassification Error')
plt.title('Training and Test Misclassification Errors vs Number of Trees')
plt.legend()
plt.grid(True)
plt.show()


print(f"{'Number of Trees (k)':<20}{'Training Error':<20}{'Test Error':<20}")
for i in range(len(k)):
    print(f"{k[i]:<20}{train_error[i]:<20}{test_error[i]:<20}")
# %%
