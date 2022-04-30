import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = [], np.zeros(iterations), np.zeros(iterations)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        sign = 1
        self.D_ = np.zeros((self.iterations_, X.shape[0]))
        self.D_[0] = np.full((X.shape[0]), fill_value=1 / X.shape[0])
        self.weights_[0] = 1
        for t in range(self.iterations_):
            # Find best weak classifier
            best_stump = self.wl_.fit(X, y)
            best_stump._find_threshold(X[:, best_stump.j_], y, sign)
            self.models_ = np.append(self.models_, best_stump)
            # Compute Error rate
            error = self.partial_loss(X, y, t)
            # Assigning classifier weight
            classifier_weight = 0.5 * np.log((1 / error) - 1)
            self.weights_[t] = classifier_weight
            # Update sample weight
            self.D_[t + 1] = self.D_[t] * np.exp(-y * self.weights_[t] * best_stump.predict(X))
            # Normalize sample weight
            self.D_[t + 1] = self.D_[t + 1] / np.sum(self.D_[t + 1])

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        sum = np.zeros(X.shape[0])
        for i in range(self.iterations_):
            sum += self.weights_[i] * self.models_[i].predict(X)
        return np.sign(sum)

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
        labels = self._predict(X)
        return np.sum((y != labels))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        sum = np.zeros(X.shape[0])
        for t in range(T+1):
            sum += self.weights_[t] * self.models_[t].predict(X)
        return np.sign(sum)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        labels = self.partial_predict(X, T)
        return np.sum(self.D_[T] * (y != labels))
