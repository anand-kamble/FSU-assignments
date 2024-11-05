#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

# Create 'plots' directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Parameters
sigma = 3
a_values = [0, 1, 2, 3, 4]
num_runs = 10
num_samples = 500

# Lists to store results
a_all = []
kmeans_accuracies_all = []
em_accuracies_all = []
kmeans_ari_all = []
em_ari_all = []

for a in a_values:
    # Generate X_Q: 500 observations from N(0, sigma^2 * I)
    X_Q = np.random.normal(loc=0, scale=sigma, size=(num_samples, 2))
    y_Q = np.zeros(num_samples, dtype=int)  # Labels for X_Q

    # Generate X_a: 500 observations from N((a, 0), I)
    mu = np.array([a, 0])
    X_a = np.random.normal(loc=mu, scale=1.0, size=(num_samples, 2))
    y_a = np.ones(num_samples, dtype=int)  # Labels for X_a

    # Merge datasets
    X = np.vstack((X_Q, X_a))
    y_true = np.hstack((y_Q, y_a))

    # For plotting clustering results when a=0
    if a == 0:
        X_plot = X.copy()
        y_true_plot = y_true.copy()

    for run in range(num_runs):
        # Set random state for reproducibility
        random_state = run

        # K-Means clustering
        kmeans = KMeans(n_clusters=2, n_init=1, random_state=random_state)
        y_kmeans = kmeans.fit_predict(X)

        # EM clustering using Gaussian Mixture Models
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=random_state)
        y_em = gmm.fit_predict(X)

        # Function to compute accuracy
        def compute_accuracy(y_true, y_pred):
            cm = contingency_matrix(y_true, y_pred)
            # Solve the linear sum assignment problem (maximize diagonal sum)
            row_ind, col_ind = linear_sum_assignment(-cm)
            accuracy = cm[row_ind, col_ind].sum() / cm.sum()
            return accuracy

        # Compute accuracy and ARI for K-Means
        kmeans_accuracy = compute_accuracy(y_true, y_kmeans)
        kmeans_ari = adjusted_rand_score(y_true, y_kmeans)
        kmeans_accuracies_all.append(kmeans_accuracy)
        kmeans_ari_all.append(kmeans_ari)

        # Compute accuracy and ARI for EM
        em_accuracy = compute_accuracy(y_true, y_em)
        em_ari = adjusted_rand_score(y_true, y_em)
        em_accuracies_all.append(em_accuracy)
        em_ari_all.append(em_ari)

        # Store the value of 'a' for plotting
        a_all.append(a)

        # Save clustering results for plotting when a=0 and first run
        if a == 0 and run == 0:
            y_kmeans_plot = y_kmeans.copy()
            y_em_plot = y_em.copy()

# Plot clustering results for a=0
plt.figure(figsize=(8, 6))
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_kmeans_plot, cmap='viridis', s=10)
plt.title('K-Means Clustering Result for a=0')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('plots/kmeans_clustering_a0.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_em_plot, cmap='viridis', s=10)
plt.title('EM Clustering Result for a=0')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('plots/em_clustering_a0.png')
plt.close()

# Plot Accuracy vs. 'a' for all runs
plt.figure(figsize=(8, 6))
plt.scatter(a_all, kmeans_accuracies_all, color='red', label='K-Means')
plt.scatter(a_all, em_accuracies_all, color='black', label='EM')
plt.title('Accuracy vs. a')
plt.xlabel('a')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('plots/accuracy_vs_a.png')
plt.close()

# Plot Adjusted Rand Index vs. 'a' for all runs
plt.figure(figsize=(8, 6))
plt.scatter(a_all, kmeans_ari_all, color='red', label='K-Means')
plt.scatter(a_all, em_ari_all, color='black', label='EM')
plt.title('Adjusted Rand Index vs. a')
plt.xlabel('a')
plt.ylabel('Adjusted Rand Index')
plt.legend()
plt.savefig('plots/ari_vs_a.png')
plt.close()


#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from numpy.linalg import inv

# Create 'plots' directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Parameters
num_runs = 10
num_samples = 500

# Lists to store results
kl_divergences = []
kmeans_iso_accuracies = []
kmeans_full_accuracies = []
em_accuracies = []
kmeans_iso_ari = []
kmeans_full_ari = []
em_ari = []

for run in range(num_runs):
    # Generate random rotation matrix U
    M = np.random.normal(0, 1, (2, 2))
    U, _, _ = np.linalg.svd(M)
    
    # Covariance matrix Σ = U * diag(100, 1) * U^T
    D = np.diag([100, 1])
    Sigma = U @ D @ U.T

    # Generate datasets
    X_Q = np.random.multivariate_normal(mean=[0, 0], cov=Sigma, size=num_samples)
    X_P = np.random.multivariate_normal(mean=[10, 0], cov=Sigma, size=num_samples)
    X = np.vstack((X_Q, X_P))
    y_true = np.hstack((np.zeros(num_samples), np.ones(num_samples)))

    # Compute KL divergence D_KL(P || Q)
    def kl_divergence(mu1, Sigma1, mu2, Sigma2):
        d = len(mu1)
        inv_Sigma2 = inv(Sigma2)
        term1 = np.log(np.linalg.det(Sigma2) / np.linalg.det(Sigma1))
        term2 = np.trace(inv_Sigma2 @ Sigma1)
        term3 = (mu2 - mu1).T @ inv_Sigma2 @ (mu2 - mu1)
        return 0.5 * (term1 - d + term2 + term3)

    mu_P = np.array([10, 0])
    mu_Q = np.array([0, 0])
    D_KL = kl_divergence(mu_P, Sigma, mu_Q, Sigma)
    kl_divergences.append(D_KL)

    # K-Means with isotropic covariance (default K-Means)
    kmeans_iso = KMeans(n_clusters=2, n_init=10, random_state=run)
    y_kmeans_iso = kmeans_iso.fit_predict(X)

    # K-Means with full covariance (simulated using GaussianMixture with 'spherical' covariance)
    kmeans_full = GaussianMixture(n_components=2, covariance_type='spherical', random_state=run)
    y_kmeans_full = kmeans_full.fit_predict(X)

    # EM clustering
    gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=1, random_state=run)
    y_em = gmm.fit_predict(X)

    # Compute accuracies and ARI
    def compute_accuracy(y_true, y_pred):
        cm = contingency_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(-cm)
        accuracy = cm[row_ind, col_ind].sum() / cm.sum()
        return accuracy

    # K-Means isotropic
    acc_iso = compute_accuracy(y_true, y_kmeans_iso)
    ari_iso = adjusted_rand_score(y_true, y_kmeans_iso)
    kmeans_iso_accuracies.append(acc_iso)
    kmeans_iso_ari.append(ari_iso)

    # K-Means full covariance (simulated)
    acc_full = compute_accuracy(y_true, y_kmeans_full)
    ari_full = adjusted_rand_score(y_true, y_kmeans_full)
    kmeans_full_accuracies.append(acc_full)
    kmeans_full_ari.append(ari_full)

    # EM
    acc_em = compute_accuracy(y_true, y_em)
    ari_em = adjusted_rand_score(y_true, y_em)
    em_accuracies.append(acc_em)
    em_ari.append(ari_em)

    # Plot clustering results for the first four runs
    if run < 4:
        # K-Means Isotropic
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans_iso, cmap='viridis', s=10)
        plt.title(f'Run {run+1}: K-Means (Isotropic)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig(f'plots/kmeans_iso_run{run+1}.png')
        plt.close()

        # K-Means Full Covariance (simulated)
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans_full, cmap='viridis', s=10)
        plt.title(f'Run {run+1}: K-Means (Full Covariance)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig(f'plots/kmeans_full_run{run+1}.png')
        plt.close()

        # EM Clustering
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y_em, cmap='viridis', s=10)
        plt.title(f'Run {run+1}: EM Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig(f'plots/em_clustering_run{run+1}.png')
        plt.close()

# Plot Accuracy vs. KL Divergence
plt.figure(figsize=(8, 6))
plt.scatter(kl_divergences, kmeans_iso_accuracies, color='red', label='K-Means Isotropic')
plt.scatter(kl_divergences, kmeans_full_accuracies, color='green', label='K-Means Full Covariance')
plt.scatter(kl_divergences, em_accuracies, color='blue', label='EM')
plt.title('Accuracy vs. KL Divergence')
plt.xlabel('KL Divergence D_KL(P || Q)')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('plots/accuracy_vs_kl.png')
plt.close()

# Plot Adjusted Rand Index vs. KL Divergence
plt.figure(figsize=(8, 6))
plt.scatter(kl_divergences, kmeans_iso_ari, color='red', label='K-Means Isotropic')
plt.scatter(kl_divergences, kmeans_full_ari, color='green', label='K-Means Full Covariance')
plt.scatter(kl_divergences, em_ari, color='blue', label='EM')
plt.title('Adjusted Rand Index vs. KL Divergence')
plt.xlabel('KL Divergence D_KL(P || Q)')
plt.ylabel('Adjusted Rand Index')
plt.legend()
plt.savefig('plots/ari_vs_kl.png')
plt.close()

# Create a table of results
import pandas as pd

results_df = pd.DataFrame({
    'Run': np.arange(1, num_runs + 1),
    'KL Divergence': kl_divergences,
    'K-Means Iso Acc': kmeans_iso_accuracies,
    'K-Means Iso ARI': kmeans_iso_ari,
    'K-Means Full Acc': kmeans_full_accuracies,
    'K-Means Full ARI': kmeans_full_ari,
    'EM Acc': em_accuracies,
    'EM ARI': em_ari
})

# Save the results table to a CSV file
results_df.to_csv('plots/clustering_results.csv', index=False)

# Optionally, print the DataFrame
print(results_df)
