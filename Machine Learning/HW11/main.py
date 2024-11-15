# %%
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob

horse_images_path: list[str] = sorted(glob.glob('horses/*.png'))
bird_image_path = 'bird.png'

horses: list[np.ndarray] | np.ndarray = []
for img_path in horse_images_path:
    img: Image.Image = Image.open(img_path).convert('L')
    # Normalize pixel values to [0,1]
    imgArray = np.asarray(img, dtype=float) / 255.0
    horses.append(imgArray.flatten())
horses = np.array(horses)  # shape: (327, 128*128)

bird_img: Image.Image = Image.open(bird_image_path).convert('L')
bird_array = np.asarray(bird_img, dtype=float) / 255.0
bird_vector:np.ndarray = bird_array.flatten()  # Flatten bird image

# 2
mean_horse:np.ndarray = np.mean(horses, axis=0)

# 3
horses_centered = horses - mean_horse

# 4
U, S, VT = np.linalg.svd(horses_centered, full_matrices=False)
V = VT.T  # Principal components (eigenvectors)

# a) 
remainingSingularValues:np.ndarray = S[2:]
plt.figure(figsize=(6, 4))
plt.plot(remainingSingularValues, marker='o')
plt.title('Remaining Singular Values (Sorted in Decreasing Order)')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.savefig('remaining_singular_values.png')
plt.show()

# b) 
pcs:np.ndarray = V[:, :2]  
HorseProjections:np.ndarray = horses_centered.dot(pcs)  # shape: (327, 2)

plt.figure(figsize=(6, 4))
plt.scatter(HorseProjections[:, 0],
            HorseProjections[:, 1], c='black', label='Horses')
plt.title('Projection of Horses onto First 2 PCs')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.savefig('horse_projections.png')  # Save plot
plt.show()

# c) 
bird_centered = bird_vector - mean_horse
BirdProjection:np.ndarray = bird_centered.dot(pcs)

plt.figure(figsize=(6, 4))
plt.scatter(HorseProjections[:, 0],
            HorseProjections[:, 1], c='black', label='Horses')
plt.scatter(BirdProjection[0], BirdProjection[1],
            c='red', marker='x', s=100, label='Bird')
plt.title('Horse Projections and Bird Projection on First 2 PCs')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.savefig('horse_bird_projection.png')  # Save plot
plt.show()

# d) 
horse070_img: Image.Image = Image.open('horses/horse070.png').convert('L')
horse070_array = np.asarray(horse070_img, dtype=float) / 255.0
horse070_vector:np.ndarray = horse070_array.flatten()

horse070_centered = horse070_vector - mean_horse

num_pcs = 32
pcs_32:np.ndarray = V[:, :num_pcs]
scores_32:np.ndarray = horse070_centered.dot(pcs_32)
horse070_reconstructed:np.ndarray = mean_horse + scores_32.dot(pcs_32.T)
horse070_reconstructed_binary:np.ndarray = (
    horse070_reconstructed > 0.5).astype(float)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(horse070_array.reshape(128, 128), cmap='gray')
axs[0].axis('off')
axs[0].set_title('Original Horse070')
axs[1].set_title('Binary Reconstruction (32 PCs)')
axs[1].imshow(horse070_reconstructed_binary.reshape(128, 128), cmap='gray')
axs[1].axis('off')
plt.savefig('horse070_reconstruction.png')  # Save figure
plt.show()

# e) 
bird_reconstructed:np.ndarray = mean_horse + (bird_centered.dot(pcs_32)).dot(pcs_32.T)
bird_reconstructed_binary:np.ndarray = (bird_reconstructed > 0.5).astype(float)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(bird_array.reshape(128, 128), cmap='gray')
axs[1].imshow(bird_reconstructed_binary.reshape(128, 128), cmap='gray')
axs[0].set_title('Original Bird')
axs[1].set_title('Binary Reconstruction (32 PCs)')
axs[0].axis('off')
axs[1].axis('off')
plt.savefig('bird_reconstruction.png')  # Save figure
plt.show()

# f)
horse_reconstructions_32 = mean_horse + \
    (horses_centered.dot(pcs_32)).dot(pcs_32.T)
distances_horses = np.linalg.norm(
    horses_centered - (horse_reconstructions_32 - mean_horse), axis=1)
distance_bird = np.linalg.norm(
    bird_centered - (bird_reconstructed - mean_horse))

plt.figure(figsize=(6, 4))
plt.scatter(HorseProjections[:, 1],
            distances_horses, c='black', label='Horses')
plt.scatter(BirdProjection[1], distance_bird,
            c='red', marker='x', s=100, label='Bird')
plt.title('Distances vs. Projection on 2nd PC')
plt.ylabel('Distance to 32-PC Plane')
plt.xlabel('Projection on PC2')
plt.legend()
plt.savefig('distances_vs_pc2.png')  # Save plot
plt.show()

# g)
plt.figure(figsize=(6, 4))
plt.hist(distances_horses, bins=20, color='black')
plt.title('Histogram of Distances to 32-PC Plane for Horses')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.savefig('hist_distances_horses.png')  # Save plot
plt.show()

# %%
