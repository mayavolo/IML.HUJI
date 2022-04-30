from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        information_gains = np.zeros(X.shape[1])
        y_unique, y_counts = np.unique(y, return_counts=True)
        entropy = -np.sum((y_counts / y.shape[0]) * np.log2(y_counts / y.shape[0]))

        for f_index in range(X.shape[1]):
            unique, counts = np.unique(X[:, f_index], return_counts=True)
            counts = np.sum(counts / X.shape[0])
            information_gains[f_index] = entropy - counts
        self.j_ = np.argmax(information_gains)  # GAINS GAINS GAINS

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        minus_sign_indices = (X[:, self.j_] < self.threshold_).nonzero()[0]
        plus_sign_indices = (X[:, self.j_] >= self.threshold_).nonzero()[0]
        y = np.zeros(X.shape[0])
        y[minus_sign_indices] = -self.sign_
        y[plus_sign_indices] = self.sign_
        return y

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> float:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        self.sign_ = sign
        unique_vals = np.unique(values)
        losses = np.zeros((unique_vals.shape[0]))
        for i in range(unique_vals.shape[0]):
            minus_sign_indices = (values < unique_vals[i]).nonzero()[0]
            plus_sign_indices = (values >= unique_vals[i]).nonzero()[0]
            y_pred = np.zeros(values.shape[0])
            y_pred[minus_sign_indices] = -sign
            y_pred[plus_sign_indices] = sign
            losses[i] = np.sum(labels != y_pred)
        self.threshold_ = unique_vals[np.argmin(losses)]
        return losses[np.argmin(losses)] / values.shape[0]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        predicted_y = self.predict(X)
        return np.sum(y != predicted_y) / X.shape[0]
