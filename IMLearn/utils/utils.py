from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .25) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    # Creating train data frames
    train_x_df = pd.DataFrame()
    train_y_df = pd.Series(dtype='float64')

    # Number of samples to keep from the data frame.
    number_of_samples = np.ceil(X.shape[0] * train_proportion).astype(int)

    # Sampling random rows from X
    x_sampled_rows = X.sample(n=number_of_samples)
    sampled_row_indices = x_sampled_rows.index
    y_sampled_labels = y.loc[sampled_row_indices]

    # Dropping the samples rows and labels from the data frames
    X = X.drop(sampled_row_indices)
    y = y.drop(sampled_row_indices)

    # Adding the sampled rows to the train data frames
    train_x_df = pd.concat([train_x_df, x_sampled_rows])
    train_y_df = pd.concat([train_y_df, y_sampled_labels])

    return train_x_df, train_y_df, X, y


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
