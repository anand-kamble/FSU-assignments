# %%
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans


def spectral_clustering(image_path, k, sigma=0.1):
    # Load image and normalize
    image: np.ndarray = io.imread(image_path) / 255.0
    height: int = image.shape[0]
    width: int = image.shape[1]
    channels: int = image.shape[2]
    n_pixels: int = height * width

    # Construct the affinity matrix A
    data: list = []
    rows: list = []
    cols: list = []

    for i in range(height):
        for j in range(width):
            idx: int = i * width + j
            I_i: np.ndarray = image[i, j, :]

            # Neighbor positions (left, right, up, down)
            neighbors: list = []
            if j > 0:
                neighbors.append((i, j - 1))
            if j < width - 1:
                neighbors.append((i, j + 1))
            if i > 0:
                neighbors.append((i - 1, j))
            if i < height - 1:
                neighbors.append((i + 1, j))

            for ni, nj in neighbors:
                idx_neighbor: int = ni * width + nj
                I_j: np.ndarray = image[ni, nj, :]
                weight: float = np.exp(-np.sum((I_i - I_j) ** 2) / sigma ** 2)
                data.append(weight)
                rows.append(idx)
                cols.append(idx_neighbor)

    # Build the sparse affinity matrix A
    A = sp.coo_matrix((data, (rows, cols)), shape=(n_pixels, n_pixels))
    A = (A + A.transpose()) / 2  # Ensure symmetry

    # Compute the normalized Laplacian
    degrees: np.ndarray = np.array(A.sum(axis=1)).flatten()
    degrees_sqrt_inv = 1.0 / np.sqrt(degrees)
    # Handle divisions by zero
    degrees_sqrt_inv[np.isinf(degrees_sqrt_inv)] = 0
    D_sqrt_inv = sp.diags(degrees_sqrt_inv)
    L_sym = sp.eye(n_pixels) - D_sqrt_inv.dot(A).dot(D_sqrt_inv)

    # Compute the first k+1 eigenvectors
    eigenvalues, eigenvectors = eigsh(L_sym, k=k+1, which='SM')
    eigenvectors = eigenvectors[:, 1:k+1]  # Skip the first eigenvector

    # Perform k-means clustering
    kmeans: KMeans = KMeans(n_clusters=k, n_init=10)
    labels: np.ndarray = kmeans.fit_predict(eigenvectors)
    label_image: np.ndarray = labels.reshape((height, width))

    # Part a) Display the label image
    plt.imshow(label_image, cmap='nipy_spectral')
    plt.title(f'Spectral Clustering with {k} Clusters')
    plt.axis('off')
    plt.show()

    # Part b) Construct the clustered image
    flat_image: np.ndarray = image.reshape(-1, 3)
    clustered_image: np.ndarray = np.zeros_like(flat_image)

    for cluster in range(k):
        mask = (labels == cluster)
        cluster_pixels = flat_image[mask]
        mean_color = cluster_pixels.mean(axis=0)
        clustered_image[mask] = mean_color

    clustered_image = clustered_image.reshape((height, width, 3))

    # Display the clustered image
    plt.imshow(clustered_image)
    plt.title(f'Clustered Image with {k} Clusters')
    plt.axis('off')
    plt.show()


# Run spectral clustering with 15 clusters
spectral_clustering('scene2.jpg', k=15)

# Run spectral clustering with 25 clusters
spectral_clustering('scene2.jpg', k=25)

# %%
