"""
=================================================
Dataset Loader for Machine Learning Models
Created by: Anand Kamble
Email: anandmk837@gmail.com
Date: 2024-10-04
Description: This script loads specific datasets ('Gisette', 'dexter', 'Madelon') 
from local paths for training and testing machine learning models. The data is 
loaded using numpy and returns both the training and test sets along with their 
respective labels.
=================================================
"""


import os
from pathlib import Path
import re
from typing import Literal
import numpy as np
import numpy.typing as npt
from sklearn import base


def get_dataset_dir_path() -> Path:
    """
    Returns the absolute path of the project's dataset folder.

    This function is useful when handling paths for loading datasets 
    that reside within the project directory.

    Returns:
    --------
    Path
        Absolute path of the dataset directory.

    Example:
    --------
    >>> get_dataset_dir_path()
    PosixPath('/home/anandkamble/my_project/dataset')
    """
    return Path(__file__).parent.absolute()


def load_dataset(DATASET: Literal['Gisette', 'dexter', 'Madelon']) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Loads the specified dataset and returns the training and testing data, 
    along with their corresponding labels.

    Parameters:
    -----------
    DATASET : Literal['Gisette', 'dexter', 'Madelon']
        The name of the dataset to load. Must be one of the following:
        'Gisette', 'dexter', 'Madelon'.

    Returns:
    --------
    `tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]] `   
        A tuple containing:
        - X_train: Training data (numpy array)
        - y_train: Training labels (numpy array)
        - X_test: Testing data (numpy array)
        - y_test: Testing labels (numpy array)

    Raises:
    -------
    ValueError
        If the DATASET argument is not one of 'Gisette', 'dexter', or 'Madelon'.

    Example:
    --------
    >>> X_train, y_train, X_test, y_test = load_dataset('Gisette')
    >>> X_train.shape
    (6000, 5000)
    >>> y_train.shape
    (6000,)

    >>> X_train, y_train, X_test, y_test = load_dataset('dexter')
    >>> X_train.shape
    (300, 20000)

    >>> X_train, y_train, X_test, y_test = load_dataset('Madelon')
    >>> X_train.shape
    (2000, 500)
    """

    y_train: npt.NDArray[np.float64] | None = None
    X_train: npt.NDArray[np.float64] | None = None
    X_test: npt.NDArray[np.float64] | None = None
    y_test: npt.NDArray[np.float64] | None = None

    if DATASET not in ['Gisette', 'dexter', 'Madelon']:
        raise ValueError(
            'DATASET must be one of "Gisette", "dexter", "Madelon"')

    base_path: Path = get_dataset_dir_path()

    match DATASET:
        case 'Gisette':
            train_data_path = base_path.joinpath(
                'Gisette', 'gisette_train.data')
            train_labels_path = base_path.joinpath(
                'Gisette', 'gisette_train.labels')
            test_data_path = base_path.joinpath(
                'Gisette', 'gisette_valid.data')
            test_labels_path = base_path.joinpath(
                'Gisette', 'gisette_valid.labels')

            print(train_data_path)
            X_train = np.loadtxt(train_data_path)
            y_train = np.loadtxt(train_labels_path)
            X_test = np.loadtxt(test_data_path)
            y_test = np.loadtxt(test_labels_path)

        case 'dexter':
            train_data_path = base_path.joinpath('dexter', 'dexter_train.csv')
            train_labels_path = base_path.joinpath(
                'dexter', 'dexter_train.labels')
            test_data_path = base_path.joinpath('dexter', 'dexter_valid.csv')
            test_labels_path = base_path.joinpath(
                'dexter', 'dexter_valid.labels')

            X_train = np.loadtxt(train_data_path, delimiter=',')
            y_train = np.loadtxt(train_labels_path)
            X_test = np.loadtxt(test_data_path, delimiter=',')
            y_test = np.loadtxt(test_labels_path)

        case 'Madelon':
            train_data_path = base_path.joinpath(
                'MADELON', 'madelon_train.data')
            train_labels_path = base_path.joinpath(
                'MADELON', 'madelon_train.labels')
            test_data_path = base_path.joinpath(
                'MADELON', 'madelon_valid.data')
            test_labels_path = base_path.joinpath(
                'MADELON', 'madelon_valid.labels')

            X_train = np.loadtxt(train_data_path)
            y_train = np.loadtxt(train_labels_path)
            X_test = np.loadtxt(test_data_path)
            y_test = np.loadtxt(test_labels_path)

    return X_train, y_train, X_test, y_test
