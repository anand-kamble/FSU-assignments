import numpy as np
import matplotlib.pyplot as plt

# Function to load 2D array from text file
def load_array_from_txt(filename):
    return np.loadtxt(filename)

# Number of files
num_files = 12  # Update with the actual number of files you have

# List to store arrays
arrays = []

# Load arrays from files
for i in range(num_files):
    filename = f"solution_{i}.txt"
    arrays.append(load_array_from_txt(filename))

# Create a heatmap
fig, ax = plt.subplots()

# Plot each array
for i, array in enumerate(arrays):
    ax.imshow(array, cmap='hot', interpolation='nearest', alpha=0.5, label=f'Data {i+1}')


plt.title('Heatmap from Multiple Data Files')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()
