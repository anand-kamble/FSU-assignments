#%%
"""
===================================================================
Neural Network for Grayscale Image Reconstruction Using Pixel Values
===================================================================
Author  : Anand Kamble
Date    : 17th October 2024
Course  : Applied Machine Learning, Homework 7
Institution: Florida State University, Department of Scientific Computing

Description:
------------
This script trains a fully connected neural network to reconstruct a grayscale image ('horse033b.png') 
based on its pixel coordinates and pixel intensity values. The task involves designing multiple 
neural networks with varying depths and complexity, training them using stochastic gradient descent 
(SGD), and visualizing the reconstruction performance across epochs.

Key Features:
-------------
- Four models (Net_a to Net_d) with different architectures ranging from 1 to 4 hidden layers.
- Input features include standardized (x, y) pixel coordinates.
- The output is the pixel intensity value, centered to have zero mean.
- Training performed using Mean Squared Error (MSE) loss with SGD optimizer.
- Learning rate scheduler reduces the learning rate every 100 epochs for smooth convergence.
- Reconstructed image visualized after training using the trained model's predictions.

Dependencies:
-------------
- Python 3.11
- PyTorch
- NumPy
- Matplotlib
- tqdm (for progress bars)

Ensure the image 'horse033b.png' is located in the same directory as this script before running it.

Usage:
------
Run this script in an environment with the necessary dependencies installed. The training process will
execute for 300 epochs, and both the loss function plot and the reconstructed image will be displayed 
at the end of the training.

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Load the image
# Ensure 'horse033b.png' is in the same directory as this script
image = Image.open('horse033b.png').convert('L')  # Convert to grayscale
image = np.array(image)

# Get image dimensions (height x width)
height, width = image.shape  # Should be 93 x 128

# Create coordinate grid
x_coords = np.arange(width)
y_coords = np.arange(height)
xx, yy = np.meshgrid(x_coords, y_coords)

# Flatten the arrays and stack to create input coordinates
inputs = np.stack([xx.flatten(), yy.flatten()], axis=1)
targets = image.flatten().astype(np.float32)

# Standardize inputs (zero mean and unit variance)
inputs_mean = inputs.mean(axis=0)
inputs_std = inputs.std(axis=0)
inputs_standardized = (inputs - inputs_mean) / inputs_std

# Center targets to have zero mean
targets_mean = targets.mean()
targets_standardized = targets - targets_mean

# Convert to PyTorch tensors
inputs_tensor = torch.from_numpy(inputs_standardized).float()
targets_tensor = torch.from_numpy(targets_standardized).float().unsqueeze(1)  # (n_samples, 1)

# Create dataset and dataloader
dataset = TensorDataset(inputs_tensor, targets_tensor)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the neural network
class Net_a(nn.Module):
    def __init__(self):
        super(Net_a, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Net_b(nn.Module):
    def __init__(self):
        super(Net_b, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
class Net_c(nn.Module):
    def __init__(self):
        super(Net_c, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x
    
class Net_d(nn.Module):
    def __init__(self):
        super(Net_d, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 128)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x


# Initialize the network
net = Net_d()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

net.apply(initialize_weights)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# Training loop
num_epochs = 300
loss_list = []
pbar = tqdm(range(1, num_epochs + 1))
for epoch in pbar:
    running_loss = 0.0
    for inputs_batch, targets_batch in data_loader:
        inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs_batch)
        loss = criterion(outputs, targets_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs_batch.size(0)
        
    epoch_loss = running_loss / len(dataset)
    loss_list.append(epoch_loss)
    
    scheduler.step()
    
    pbar.set_description(f"Loss: {epoch_loss:.6f}")

# Plot the loss function vs epoch number
plt.figure()
plt.plot(range(1, num_epochs + 1), loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Reconstruct the image from the trained network
# Create grid of all (x, y)
xx, yy = np.meshgrid(np.arange(width), np.arange(height))
grid_inputs = np.stack([xx.flatten(), yy.flatten()], axis=1)

# Standardize inputs
grid_inputs_standardized = (grid_inputs - inputs_mean) / inputs_std

# Convert to tensor
grid_inputs_tensor = torch.from_numpy(grid_inputs_standardized).float().to(device)

# Pass through network
with torch.no_grad():
    outputs = net(grid_inputs_tensor).cpu().numpy().flatten()

# Add back the mean to the outputs
outputs = outputs + targets_mean

# Reshape to original image dimensions
reconstructed_image = outputs.reshape((height, width))

# Display the reconstructed image
plt.figure()
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')
plt.show()

