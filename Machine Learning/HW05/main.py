# %%
import sys
sys.path.append('/home/amk23j/Documents/FSU-assignments/Machine Learning/')
from dataset import get_dataset_dir_path, load_dataset
from typing import Literal
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import os

# %%
class FSA:
    def __init__(self, k: int, mu=300, s=0.001, Niter=300, lr=0.01, h=0.1):
        self.mu: int = mu
        self.k: int = k
        self.Niter: int = Niter
        self.lr: float = lr
        self.s: float = s
        self.h: float = h

    def predict(self, X:torch.Tensor) -> torch.Tensor:
        xw: torch.Tensor = X[:, self.idx].float() @ self.w.view(-1, 1) + self.w0
        return xw

    def fit(self, X:torch.Tensor, y:torch.Tensor, device:torch.device):
        p = X.shape[1]
        y = y.clone()
        y[y == 0] = -1
        self.idx = torch.arange(0, p).long().to(device)
        self.w = torch.zeros((p, 1), device=device, requires_grad=True)
        self.w0 = torch.zeros(1, device=device, requires_grad=True)
        optimizer = optim.SGD([self.w, self.w0], lr=self.lr)
        losses = []
        for i in range(1, self.Niter + 1):
            optimizer.zero_grad()
            xw = self.predict(X)
            z = y * xw.squeeze()
            loss = torch.zeros_like(z)
            idx1 = (z >= 1 + self.h)
            idx2: torch.Tensor = (torch.abs(1 - z) <= self.h)
            idx3 = (z <= 1 - self.h)
            loss[idx1] = 0
            loss[idx2] = ((1 + self.h - z[idx2]) ** 2) / (4 * self.h)
            loss[idx3] = 1 - z[idx3]
            loss1 = torch.mean(loss) + self.s * \
                torch.sum(self.w ** 2) + self.s * self.w0 ** 2
            loss1.backward()
            optimizer.step()
            m = int(self.k + (p - self.k) * max(0,
                    (self.Niter - 2 * i) / (2 * i * self.mu + self.Niter)))
            if m < self.w.shape[0]:
                sw = torch.abs(self.w.view(-1))
                sw_sorted, indices = torch.sort(sw, descending=True)
                thr = sw_sorted[m - 1].item()
                j = torch.where(sw >= thr)[0]
                self.idx = self.idx[j]
                self.w = self.w[j].detach().clone().requires_grad_(True)
                optimizer = optim.SGD([self.w, self.w0], lr=self.lr)
            losses.append(loss1.item())
        return losses

# %%


def main(DATASET: Literal['Gisette', 'dexter', 'Madelon']):
    # Create output directory
    output_dir = f'output_{DATASET}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load training and test data
    X_train, y_train, X_test, y_test = load_dataset(DATASET)
    # Normalize the data
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    X_train_norm = (X_train - mean) / std
    # Use training mean and std for test data
    X_test_norm = (X_test - mean) / std

    # Convert to torch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_tensor: torch.Tensor = torch.tensor(X_train_norm).float().to(device)
    y_train_tensor: torch.Tensor = torch.tensor(y_train).float().to(device)
    X_test_tensor: torch.Tensor = torch.tensor(X_test_norm).float().to(device)
    y_test_tensor: torch.Tensor = torch.tensor(y_test).float().to(device)

    k_list: list[int] = [10, 30, 100, 300, 500]
    train_errors: list[float] = []
    test_errors: list[float] = []
    train_losses_k30 = None
    model_k30 = None

    for k in k_list:
        model = FSA(k=k, mu=300, s=0.001, Niter=300, lr=0.01, h=0.1)
        losses = model.fit(X_train_tensor, y_train_tensor.clone(), device)
        # For k=30, save the training loss vs iteration
        if k == 30:
            train_losses_k30 = losses
            model_k30 = model  # Save the model trained for k=30
            # Plot training loss vs iteration number for k=30
            plt.figure()
            plt.plot(range(1, len(losses)+1), losses)
            plt.xlabel('Iteration')
            plt.ylabel('Training Loss')
            plt.title(f'Training Loss vs Iteration (k=30) on {DATASET}')
            plt.savefig(os.path.join(
                output_dir, f'Training_Loss_k30_{DATASET}.png'))
            plt.close()
        # Evaluate on training set
        with torch.no_grad():
            y_train_pred = model.predict(X_train_tensor)
            y_train_pred_labels = (y_train_pred.squeeze()
                                   >= 0).cpu().numpy().astype(int)
            y_train_true_labels = y_train_tensor.cpu().numpy()
            y_train_true_labels[y_train_true_labels == -1] = 0
            train_error = np.mean(y_train_pred_labels != y_train_true_labels)
            train_errors.append(train_error)
            # Evaluate on test set
            y_test_pred = model.predict(X_test_tensor)
            y_test_pred_labels = (y_test_pred.squeeze() >=
                                  0).cpu().numpy().astype(int)
            y_test_true_labels = y_test_tensor.cpu().numpy()
            y_test_true_labels[y_test_true_labels == -1] = 0
            test_error = np.mean(y_test_pred_labels != y_test_true_labels)
            test_errors.append(test_error)
        print(f'Completed training for k={k}')

    # Print misclassification errors
    print('k\tTraining Error\tTest Error')
    for i, k in enumerate(k_list):
        print(f'{k}\t{train_errors[i]:.4f}\t\t{test_errors[i]:.4f}')

    # Save misclassification errors to CSV
    error_df = pd.DataFrame({
        'k': k_list,
        'Training Error': train_errors,
        'Test Error': test_errors
    })
    error_df.to_csv(os.path.join(
        output_dir, f'Misclassification_Errors_{DATASET}.csv'), index=False)

    # Save misclassification errors as LaTeX table
    with open(os.path.join(output_dir, f'Misclassification_Errors_{DATASET}.tex'), 'w') as f:
        f.write(error_df.to_latex(index=False, float_format="%.4f"))

    # Plot misclassification error vs k
    plt.figure()
    plt.plot(k_list, train_errors, label='Training Error', marker='o')
    plt.plot(k_list, test_errors, label='Test Error', marker='s')
    plt.xlabel('Number of Features (k)')
    plt.ylabel('Misclassification Error')
    plt.title(f'Misclassification Error vs Number of Features on {DATASET}')
    plt.legend()
    plt.savefig(os.path.join(
        output_dir, f'Misclassification_Error_vs_k_{DATASET}.png'))
    plt.close()

    # For k=30, plot ROC curves
    if model_k30 is not None:
        model = model_k30
        # On training set
        y_train_pred_scores = model.predict(
            X_train_tensor).squeeze().cpu().detach().numpy()
        y_train_true_labels = y_train_tensor.cpu().detach().numpy()
        y_train_true_labels[y_train_true_labels == -1] = 0
        fpr_train, tpr_train, _ = roc_curve(
            y_train_true_labels, y_train_pred_scores)
        roc_auc_train = auc(fpr_train, tpr_train)
        # On test set
        y_test_pred_scores = model.predict(
            X_test_tensor).squeeze().cpu().detach().numpy()
        y_test_true_labels = y_test_tensor.cpu().detach().numpy()
        y_test_true_labels[y_test_true_labels == -1] = 0
        fpr_test, tpr_test, _ = roc_curve(
            y_test_true_labels, y_test_pred_scores)
        roc_auc_test = auc(fpr_test, tpr_test)
        # Plot ROC curves
        plt.figure()
        plt.plot(fpr_train, tpr_train,
                 label='Training ROC (area = %0.2f)' % roc_auc_train)
        plt.plot(fpr_test, tpr_test, label='Test ROC (area = %0.2f)' %
                 roc_auc_test)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves (k=30) on {DATASET}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, f'ROC_Curves_k30_{DATASET}.png'))
        plt.close()


# %%
if __name__ == '__main__':
    # Part A
    main("Gisette")

    # Part B
    main("dexter")
    
    # Part C
    main("Madelon")

# %%
