#%%
import numpy as np
from numpy._typing._array_like import NDArray
import scipy.io
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Directories containing the data
trainDataDir = 'features_640'  # Replace with your path
testDataDir = 'features_val_640'  # Replace with your path

# Get list of files and sort them to ensure consistent class labels
trainFiles:list[str] = sorted(os.listdir(trainDataDir))
testFiles:list[str] = sorted(os.listdir(testDataDir))

# Initialize lists to hold data and labels
xTrainList:list = []
yTrainList:list = []

xTestList:list = []
yTestList:list = []

# Load training data
print("Loading training data...")
for idx, filename in enumerate(tqdm(trainFiles)):
    filepath: str = os.path.join(trainDataDir, filename)
    matContents:dict = scipy.io.loadmat(filepath)
    
    # Extract features
    data_keys:list = [key for key in matContents.keys() if not key.startswith('__')]
    
    features = matContents['feature']
    
    # Assign labels based on the index (from 0 to 9)
    labels = np.full(features.shape[0], idx)
    
    xTrainList.append(features)
    yTrainList.append(labels)

# Concatenate all training data
xTrain = np.vstack(xTrainList)
yTrain = np.concatenate(yTrainList)

# Load test data
print("Loading test data...")
for idx, filename in enumerate(tqdm(testFiles)):
    filepath = os.path.join(testDataDir, filename)
    matContents = scipy.io.loadmat(filepath)
    # Extract features
    data_keys:list = [key for key in matContents.keys() if not key.startswith('__')]
    # Try to extract features
    features:np.ndarray = matContents['feature']
    # Assign labels based on the index (from 0 to 9)
    labels:np.ndarray = np.full(features.shape[0], idx)
    xTestList.append(features)
    yTestList.append(labels)

# Concatenate all test data
xTest:np.ndarray = np.vstack(xTestList)
yTest:np.ndarray = np.concatenate(yTestList)

# Standardize the data
print("Standardizing data...")
scaler = StandardScaler()
xTrainScaled:np.ndarray = scaler.fit_transform(xTrain)
xTestScaled :np.ndarray= scaler.transform(xTest)  # Use the same mean and std

# Convert data to PyTorch tensors
xTrainTensor:torch.Tensor = torch.from_numpy(xTrainScaled).float()
yTrainTensor:torch.Tensor = torch.from_numpy(yTrain).long()

xTestTensor:torch.Tensor = torch.from_numpy(xTestScaled).float()
yTestTensor:torch.Tensor = torch.from_numpy(yTest).long()

# Part a) Train a linear classifier using cross-entropy loss (softmax loss)
print("Training linear classifier with cross-entropy loss...")

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

input_dim = xTrainTensor.shape[1]
num_classes = 10

model = LinearClassifier(input_dim, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
numEpochs = 30
batchSize = 64

trainDataset = torch.utils.data.TensorDataset(xTrainTensor, yTrainTensor)
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)

for epoch in range(numEpochs):
    model.train()
    runningLoss = 0.0
    for inputs, labels in trainLoader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item() * inputs.size(0)
    
    scheduler.step()
    epochLoss = runningLoss / len(trainDataset)
    print(f"Epoch {epoch+1}/{numEpochs}, Loss: {epochLoss:.4f}")

# Evaluate on training data
model.eval()
with torch.no_grad():
    outputs = model(xTrainTensor)
    _, yTrainPred = torch.max(outputs, 1)
    trainAccuracy:int | float | bool = (yTrainPred == yTrainTensor).float().mean().item()
    trainMisclassificationError:int | float = 1 - trainAccuracy

# Evaluate on test data
with torch.no_grad():
    outputs = model(xTestTensor)
    _, yTestPred = torch.max(outputs, 1)
    testAccuracy:int | float | bool = (yTestPred == yTestTensor).float().mean().item()
    testMisclassificationError:int | float = 1 - testAccuracy

print(f"Training misclassification error: {trainMisclassificationError:.4f}")
print(f"Test misclassification error: {testMisclassificationError:.4f}")

# Part b) Train PPCA models for each class
print("Training PPCA models for each class...")
q = 20  # Number of principal components
muList:list = []
wList:list = []
sigma2List:list = []

for classIdx in tqdm(range(num_classes)):
    X_class = xTrainScaled[yTrain == classIdx]
    N_k, D = X_class.shape
    
    # Compute the mean
    mu_k = np.mean(X_class, axis=0)
    
    # Center the data
    X_centered = X_class - mu_k
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Select top q components
    W_k = Vt[:q].T  # D x q
    
    # Estimate sigma^2
    residualVariances = (S[q:] ** 2) / (N_k - 1)
    sigma2K = np.mean(residualVariances)
    
    muList.append(mu_k)
    wList.append(W_k)
    sigma2List.append(sigma2K)
    
    print(f"Class {classIdx}: sigma^2 = {sigma2K:.4f}")

# Part c) Classification using Mahalanobis distance
print("Classifying data using PPCA-based classifier...")

def compute_mahalanobis_distance(x, mu_k, W_k, sigma2_k):
    diff:np.ndarray = x - mu_k  # D-dimensional
    M_inv:np.ndarray = np.linalg.inv(np.eye(W_k.shape[1]) + (W_k.T @ W_k) / sigma2_k)
    term1:np.ndarray = (diff @ diff) / sigma2_k
    term2:np.ndarray = (diff @ W_k @ M_inv @ W_k.T @ diff) / (sigma2_k ** 2)
    dist:np.ndarray = term1 - term2
    return dist

def classify_ppca(X, muList, wList, sigma2List):
    N:int = X.shape[0]
    num_classes:int = len(muList)
    y_pred:np.ndarray = np.zeros(N, dtype=int)
    
    for i in tqdm(range(N)):
        x:np.ndarray = X[i]
        distances: np.ndarray = np.zeros(num_classes)
        for k in range(num_classes):
            dist = compute_mahalanobis_distance(x, muList[k], wList[k], sigma2List[k])
            distances[k] = dist
        y_pred[i] = np.argmin(distances)
    return y_pred

# Classify training data
print("Classifying training data...")
y_train_pred_ppca = classify_ppca(xTrainScaled, muList, wList, sigma2List)
train_accuracy_ppca = accuracy_score(yTrain, y_train_pred_ppca)
train_misclassification_error_ppca = 1 - train_accuracy_ppca
print(f"PPCA-based classifier training misclassification error: {train_misclassification_error_ppca:.4f}")

# Classify test data
print("Classifying test data...")
y_test_pred_ppca = classify_ppca(xTestScaled, muList, wList, sigma2List)
test_accuracy_ppca = accuracy_score(yTest, y_test_pred_ppca)
test_misclassification_error_ppca = 1 - test_accuracy_ppca
print(f"PPCA-based classifier test misclassification error: {test_misclassification_error_ppca:.4f}")
